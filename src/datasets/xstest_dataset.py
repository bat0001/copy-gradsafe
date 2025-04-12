import pandas as pd
import os

def load_xstest_dataset(csv_path: str = "./data/xstest/xstest_v2_prompts.csv"):
  
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"xstest CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df