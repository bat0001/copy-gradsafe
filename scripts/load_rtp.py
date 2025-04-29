from datasets import load_dataset

ds = load_dataset("allenai/real-toxicity-prompts")

df = ds['train'].to_pandas()
df.to_csv("./data/realtoxicity/realtoxicity.csv", index=False)