import copy
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from find_critical_parameters import find_critical_para, load_model

def extract_features(model_id, dataset, gradient_norms_compare, minus_row, minus_col, use_critical_params=True):
    """
    Extract cosine similarity features for each prompt in the dataset
    
    Args:
        model_id: The ID of the model to use
        dataset: DataFrame containing 'user_input' and 'toxicity' columns
        gradient_norms_compare: Reference gradients for unsafe prompts
        minus_row, minus_col: Safety-critical parameter information
        use_critical_params: Whether to use safety-critical parameters
        
    Returns:
        features: List of feature vectors for each prompt
        labels: List of labels (1=toxic, 0=safe)
    """
    columns = dataset[['user_input', 'toxicity']]
    model, tokenizer = load_model(model_id)
    device = next(model.parameters()).device
    
    # Prompt templates
    sep_token, sep_token_id = tokenizer.unk_token, tokenizer.unk_token_id
    prompt = (
        f'<s>[INST] <<SYS>> {{system_prompt}} <</SYS>> {{content}} [/INST]' + 
        f'{{sep_token}} {{summary}} {{eos_token}}'
    )
    
    def apply_prompt_template(sample):
        txt = prompt.format(
            system_prompt='You are a helpful assistant. Help me with the following query: ',
            content=sample['source'],
            summary=sample['target'],
            eos_token=tokenizer.eos_token,
            sep_token=sep_token,
        )
        return {
            'text': txt,
        }
    
    # Calculate features for each prompt
    features = []
    labels = []
    
    for index, row in tqdm(columns.iterrows(), total=len(columns)):
        # Ground truth label
        labels.append(1 if row['toxicity'] == 1 else 0)
        
        # Calculate gradients
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        basic_sample = {}
        basic_sample["source"] = row['user_input']
        basic_sample["target"] = "Sure"
        d = apply_prompt_template(basic_sample)
        input_ids = tokenizer(d['text']).input_ids
        sep = input_ids.index(sep_token_id)
        
        input_ids = input_ids[:sep] + input_ids[sep+1:]
        input_ids = torch.tensor(np.array([input_ids])).to(device)
        target_ids = input_ids.clone()
        target_ids[:, :sep] = -100
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        neg_log_likelihood.backward()
        
        # Extract individual cosine similarities as features
        prompt_features = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and ("mlp" in name or "self" in name):
                grad_norm = param.grad.to(gradient_norms_compare[name].device)
                row_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, gradient_norms_compare[name], dim=1))
                col_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, gradient_norms_compare[name], dim=0))
                
                # If using safety-critical parameters, filter by importance
                if use_critical_params:
                    ref_row = minus_row[name].to(row_cos.device)
                    ref_col = minus_col[name].to(col_cos.device)
                    
                    # Extract only the safety-critical parameters
                    row_features = row_cos[ref_row > 1].cpu().tolist()
                    col_features = col_cos[ref_col > 1].cpu().tolist()
                else:
                    # Use all parameters
                    row_features = row_cos.cpu().tolist()
                    col_features = col_cos.cpu().tolist()
                    
                prompt_features.extend(row_features)
                prompt_features.extend(col_features)
        
        # Add a single aggregated feature (mean) for GradSafe-Zero compatibility
        mean_feature = sum(prompt_features) / len(prompt_features) if prompt_features else 0
        prompt_features.append(mean_feature)
        
        features.append(prompt_features)
    
    return features, labels

def gradsafe_adapt(model_id, train_df, test_df, gradient_norms_compare, minus_row, minus_col, use_critical_params=True):
    """
    Implement GradSafe-Adapt by training a logistic regression model on cosine similarity features
    
    Args:
        model_id: The ID of the model to use
        train_df: Training dataset
        test_df: Test dataset
        gradient_norms_compare: Reference gradients for unsafe prompts
        minus_row, minus_col: Safety-critical parameter information
        use_critical_params: Whether to use safety-critical parameters
        
    Returns:
        auprc: Area under precision-recall curve
        f1: F1 score
        precision: Precision score
        recall: Recall score
    """
    print(f"Extracting features for training set...")
    train_features, train_labels = extract_features(
        model_id, train_df, gradient_norms_compare, minus_row, minus_col, use_critical_params
    )
    
    print(f"Extracting features for test set...")
    test_features, test_labels = extract_features(
        model_id, test_df, gradient_norms_compare, minus_row, minus_col, use_critical_params
    )
    
    # Ensure all feature vectors have the same length by padding if necessary
    max_len = max(max(len(f) for f in train_features), max(len(f) for f in test_features))
    
    # Pad features to ensure consistent dimensions
    train_features_padded = [f + [0] * (max_len - len(f)) for f in train_features]
    test_features_padded = [f + [0] * (max_len - len(f)) for f in test_features]
    
    # Convert to numpy arrays
    X_train = np.array(train_features_padded)
    y_train = np.array(train_labels)
    X_test = np.array(test_features_padded)
    y_test = np.array(test_labels)
    
    print(f"Training logistic regression classifier with {X_train.shape[1]} features...")
    # Train a logistic regression classifier
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    classifier.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)
    
    precision_score_val = precision_score(y_test, y_pred)
    recall_score_val = recall_score(y_test, y_pred)
    f1_score_val = f1_score(y_test, y_pred)
    
    print("GradSafe-Adapt Results:")
    print(f"AUPRC: {auprc:.3f}")
    print(f"Precision: {precision_score_val:.3f}")
    print(f"Recall: {recall_score_val:.3f}")
    print(f"F1 Score: {f1_score_val:.3f}")
    
    return auprc, f1_score_val, precision_score_val, recall_score_val

def gradsafe_zero(model_id, test_df, gradient_norms_compare, minus_row, minus_col, use_critical_params=True, threshold=0.25):
    """
    Implement GradSafe-Zero by averaging cosine similarities across all safety-critical parameters
    
    Args:
        model_id: The ID of the model to use
        test_df: Test dataset
        gradient_norms_compare: Reference gradients for unsafe prompts
        minus_row, minus_col: Safety-critical parameter information
        use_critical_params: Whether to use safety-critical parameters
        threshold: Threshold for binary classification
        
    Returns:
        auprc: Area under precision-recall curve
        f1: F1 score
        precision: Precision score
        recall: Recall score
    """
    features, labels = extract_features(
        model_id, test_df, gradient_norms_compare, minus_row, minus_col, use_critical_params
    )
    
    # For GradSafe-Zero, we use the mean feature (last feature in each vector)
    scores = [feature[-1] for feature in features]
    
    # Calculate metrics
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)
    
    # Binary classification using threshold
    predictions = [1 if score >= threshold else 0 for score in scores]
    precision_score_val = precision_score(labels, predictions)
    recall_score_val = recall_score(labels, predictions)
    f1_score_val = f1_score(labels, predictions)
    
    print("GradSafe-Zero Results:")
    print(f"AUPRC: {auprc:.3f}")
    print(f"Precision: {precision_score_val:.3f}")
    print(f"Recall: {recall_score_val:.3f}")
    print(f"F1 Score: {f1_score_val:.3f}")
    
    return auprc, f1_score_val, precision_score_val, recall_score_val

def evaluate_all_methods(model_id, train_df, test_df):
    """
    Evaluate all GradSafe methods and report results
    
    Args:
        model_id: The ID of the model to use
        train_df: Training dataset
        test_df: Test dataset
    """
    print("Finding critical parameters...")
    gradient_norms_compare, minus_row_cos, minus_col_cos = find_critical_para(model_id)
    
    results = []
    
    # 1. GradSafe-Zero
    print("\nEvaluating GradSafe-Zero...")
    auprc, f1, precision, recall = gradsafe_zero(
        model_id, test_df, gradient_norms_compare, minus_row_cos, minus_col_cos, use_critical_params=True
    )
    results.append({
        'Method': 'GradSafe-Zero',
        'AUPRC': auprc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    # 2. GradSafe-Zero w/o Safety-Critical Parameters
    print("\nEvaluating GradSafe-Zero w/o Safety-Critical Parameters...")
    auprc, f1, precision, recall = gradsafe_zero(
        model_id, test_df, gradient_norms_compare, minus_row_cos, minus_col_cos, use_critical_params=False
    )
    results.append({
        'Method': 'GradSafe-Zero w/o Safety-Critical Parameters',
        'AUPRC': auprc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    # 3. GradSafe-Adapt
    print("\nEvaluating GradSafe-Adapt...")
    auprc, f1, precision, recall = gradsafe_adapt(
        model_id, train_df, test_df, gradient_norms_compare, minus_row_cos, minus_col_cos, use_critical_params=True
    )
    results.append({
        'Method': 'GradSafe-Adapt',
        'AUPRC': auprc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    # 4. GradSafe-Adapt w/o Safety-Critical Parameters
    print("\nEvaluating GradSafe-Adapt w/o Safety-Critical Parameters...")
    auprc, f1, precision, recall = gradsafe_adapt(
        model_id, train_df, test_df, gradient_norms_compare, minus_row_cos, minus_col_cos, use_critical_params=False
    )
    results.append({
        'Method': 'GradSafe-Adapt w/o Safety-Critical Parameters',
        'AUPRC': auprc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    # Print results table
    results_df = pd.DataFrame(results)
    results_df['P/R/F1'] = results_df.apply(
        lambda row: f"{row['Precision']:.3f}/{row['Recall']:.3f}/{row['F1']:.3f}", axis=1
    )
    
    print("\nResults Summary:")
    print(results_df[['Method', 'AUPRC', 'P/R/F1']])
    
    return results_df

if __name__ == "__main__":
    model_id = '/home/kusam/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590'
    
    # Load datasets
    train_df = pd.read_csv('./data/toxic-chat/toxic-chat_annotation_train.csv')
    test_df = pd.read_csv('./data/toxic-chat/toxic-chat_annotation_test.csv')
    
    # Evaluate all methods
    results = evaluate_all_methods(model_id, train_df, test_df)