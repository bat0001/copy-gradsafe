import argparse
import torch
import torch.nn as nn

from models.llama_model import load_model
from trainers.critical_parameters import find_critical_para
from trainers.evaluate_toxic import cos_sim_toxic
from trainers.evaluate_xstest import cos_sim_xstest
from datasets.toxic_chat_dataset import load_toxic_chat_dataset
from datasets.xstest_dataset import load_xstest_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="./model/Llama-3.2-3B-Instruct", help="Path or name of the model.")
    parser.add_argument("--toxic_csv", default="./data/toxic-chat/toxic-chat_annotation_test.csv")
    parser.add_argument("--xstest_csv", default="./data/xstest/xstest_v2_prompts.csv")
    args = parser.parse_args()

    # 1. Load model & tokenizer
    model, tokenizer = load_model(args.model_id)

    # 2. Find critical parameters
    gradient_norms_compare, minus_row_cos, minus_col_cos = find_critical_para(model, tokenizer)

    # 3. Evaluate on toxic dataset
    df_toxic = load_toxic_chat_dataset(args.toxic_csv)
    cos_sim_toxic(model, tokenizer, df_toxic, gradient_norms_compare, minus_row_cos, minus_col_cos)

    # 4. Evaluate on xstest
    df_xstest = load_xstest_dataset(args.xstest_csv)
    cos_sim_xstest(model, tokenizer, df_xstest, gradient_norms_compare, minus_row_cos, minus_col_cos)

if __name__ == "__main__":
    main()