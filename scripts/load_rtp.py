import os
import sys
import pandas as pd
from datasets import load_dataset

OUT_DIR = "data/realtoxicity"
OUT_CSV = os.path.join(OUT_DIR, "realtoxicity.csv")
os.makedirs(OUT_DIR, exist_ok=True)

print("→ Téléchargement allenai/real-toxicity-prompts …")
ds = load_dataset("allenai/real-toxicity-prompts", split="train")

ds_small = ds.shuffle(seed=42).select(range(5000))
df = ds_small.to_pandas()

def get_text(obj):
    return obj if isinstance(obj, str) else obj.get("text", "")

def get_score(obj):
    if isinstance(obj, dict) and "toxicity" in obj:
        return obj["toxicity"]

try:
    prompts = df["prompt"].apply(get_text)
    scores  = df["prompt"].apply(get_score)
except ValueError as e:
    print(f"❌ {e}")
    sys.exit(1)

toxicity_bin = (scores > 0.5).astype(int)

out_df = pd.DataFrame({"prompt": prompts, "toxicity": toxicity_bin})
out_df.to_csv(OUT_CSV, index=False)

print("✓ Fichier enregistré :", OUT_CSV, "—", len(out_df), "lignes")