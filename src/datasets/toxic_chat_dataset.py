import pandas as pd
import os

def load_toxic_chat_dataset(csv_path: str = "./data/toxic-chat/toxic-chat_annotation_test.csv"):
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Toxic chat CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df