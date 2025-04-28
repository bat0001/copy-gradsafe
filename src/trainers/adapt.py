import numpy as np, torch, torch.nn.functional as F, gc
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc,
    precision_score, recall_score, f1_score
)

from trainers.gradsafe import find_critical_para      
from utils.model import load_model         


def extract_features(model_id, df, ref, minus_row, minus_col, use_crit=True):
    """
    Retourne deux listes:
        - feats : [ [f1, f2, ...], ... ]  (dernière valeur = moyenne)
        - labels: [0/1, ...]
    """
    is_toxic = 'toxicity' in df.columns
    cols = df[['user_input','toxicity']] if is_toxic else df[['prompt','type']]

    model, tok, device = load_model(model_id)

    prompt_tmpl = (
        "<s>[INST] <<SYS>> {system_prompt} <</SYS>> {content} [/INST]"
        "{sep_token} {summary} {eos_token}"
    )
    sep_tok = tok.unk_token or tok.eos_token
    sep_id  = tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id

    feats, labels = [], []

    for _, row in tqdm(cols.iterrows(), total=len(cols), desc=f'feat-{model_id}'):
        if is_toxic:
            labels.append(int(row.toxicity))
            content = row.user_input
        else:
            labels.append(1 if "contrast" in row.type else 0)
            content = row.prompt

        text = prompt_tmpl.format(
            system_prompt="You are a helpful assistant. Help me with the following query:",
            content=content,
            sep_token=sep_tok,
            summary="Sure",
            eos_token=tok.eos_token,
        )

        ids = tok(text).input_ids
        sep = ids.index(sep_id) if sep_id in ids else len(ids)-1
        ids = torch.tensor([ids[:sep] + ids[sep+1:]], device=device)
        tgt = ids.clone(); tgt[:, :sep] = -100

        model.zero_grad()
        model(ids, labels=tgt).loss.backward()

        cur = []
        for name, p in model.named_parameters():
            if p.grad is None or ("mlp" not in name and "self" not in name):
                continue
            g = p.grad                           
            row_cos = ( torch.nan_to_num(
                           F.cosine_similarity(g, ref[name], dim=1)
                       ) if g.dim() >= 2 else torch.tensor([], device=g.device) )
            col_cos = torch.nan_to_num(
                           F.cosine_similarity(g, ref[name], dim=0)
                       )

            if use_crit:
                row_cos = row_cos[minus_row[name] > 1]
                col_cos = col_cos[minus_col[name] > 1]

            cur += row_cos.tolist() + col_cos.tolist()

        cur.append(sum(cur) / len(cur) if cur else 0)   
        feats.append(cur)

    torch.cuda.empty_cache(); gc.collect()
    return feats, labels

def gradsafe_zero(model_id, df, ref, minus_row, minus_col, use_crit=True):
    feats, y = extract_features(model_id, df, ref, minus_row, minus_col, use_crit)
    scores   = [f[-1] for f in feats]
    prec, rec, _ = precision_recall_curve(y, scores); auprc = auc(rec, prec)

    best = {'thr':0,'p':0,'r':0,'f1':0}
    for thr in np.linspace(0,1,101):
        pred=[1 if s>=thr else 0 for s in scores]
        if sum(pred)==0: continue
        p,r = precision_score(y,pred), recall_score(y,pred); f1=f1_score(y,pred)
        if f1>best['f1']: best.update({'thr':thr,'p':p,'r':r,'f1':f1})
    return {'method':'GS-Zero'+(' (all)' if not use_crit else ''),
            'auprc':auprc, **best}

def gradsafe_adapt(model_id, train_df, test_df, ref, minus_row, minus_col, use_crit=True):
    X_tr,y_tr = extract_features(model_id, train_df , ref, minus_row, minus_col, use_crit)
    X_te,y_te = extract_features(model_id, test_df  , ref, minus_row, minus_col, use_crit)

    max_dim   = max(max(len(f) for f in X_tr), max(len(f) for f in X_te))
    pad = lambda arr: [f+[0]*(max_dim-len(f)) for f in arr]
    X_tr,X_te = np.array(pad(X_tr)), np.array(pad(X_te))

    clf = LogisticRegression(max_iter=1000,class_weight='balanced').fit(X_tr,y_tr)
    proba = clf.predict_proba(X_te)[:,1]

    prec, rec, _ = precision_recall_curve(y_te, proba); auprc = auc(rec, prec)
    pred  = clf.predict(X_te)
    p,r,f1 = precision_score(y_te,pred), recall_score(y_te,pred), f1_score(y_te,pred)
    return {'method':'GS-Adapt'+(' (all)' if not use_crit else ''),
            'auprc':auprc,'thr':0,'p':p,'r':r,'f1':f1}

def evaluate_adapt_suite(model_id, train_df, test_df):
    ref, minus_row, minus_col = find_critical_para(model_id)
    res  = []
    res += [gradsafe_zero (model_id,test_df ,ref,minus_row,minus_col, True )]
    res += [gradsafe_zero (model_id,test_df ,ref,minus_row,minus_col, False)]
    res += [gradsafe_adapt(model_id,train_df,test_df,ref,minus_row,minus_col, True )]
    res += [gradsafe_adapt(model_id,train_df,test_df,ref,minus_row,minus_col, False)]
    return res