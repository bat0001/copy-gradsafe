import pandas as pd
from datasets import toxicchat, xstest, realtoxicity
from trainers.gradsafe import find_critical_para, eval_dataset
from utils.model import load_model

MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-Guard-3-1B",
]

for task, loader in {
    # "toxic": toxicchat.load,
    # "xstest": xstest.load,
    "realtoxicity": realtoxicity.load
}.items():

    df = loader()
    rows=[]
    for mid in MODELS:
        model, tok, device = load_model(mid)  # <- Charge UNE fois ici
        ref, minus_row, minus_col = find_critical_para(model, tok, device)
        best, auprc = eval_dataset(mid, model, tok, device, df, ref, minus_row, minus_col, task)
        
    
        rows.append({
            "model": mid.split("/")[-1],
            "precision": round(best['p'],3),
            "recall": round(best['r'],3),
            "f1": round(best['f1'],3),
            "auprc": round(auprc,3)
        })
    print(f"\n## Résultats {task}\n")
    print(pd.DataFrame(rows).to_markdown(index=False))