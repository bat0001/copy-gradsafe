import pandas as pd
from datasets import toxicchat, xstest, realtoxicity
from trainers.gradsafe import find_critical_para, eval_dataset, eval_prompting_baseline
from utils.model import load_model


MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-Guard-3-1B",
]

results_all = {}
for task, loader in {
    "realtoxicity": realtoxicity.load,
    "toxic": toxicchat.load,
    "xstest": xstest.load,
}.items():

    df = loader()
    rows = []
    for mid in MODELS:
        model, tok, device = load_model(mid)
        model.config.pad_token_id = tok.eos_token_id

        if mid.lower().endswith("llama-guard-3-1b"):
            baseline = eval_prompting_baseline(mid, df)
            best = { 'p': baseline['p'], 'r': baseline['r'], 'f1': baseline['f1'] }
            auprc = baseline['auprc']
        else:
            ref, minus_row, minus_col = find_critical_para(model, tok, device)
            best, auprc = eval_dataset(mid, model, tok, device, df, ref, minus_row, minus_col, task)

        rows.append({
            "model": mid.split("/")[-1],
            "precision": round(best['p'], 3),
            "recall":    round(best['r'], 3),
            "f1":        round(best['f1'], 3),
            "auprc":     round(auprc,    3)
        })

    results_all[task] = pd.DataFrame(rows)

for task, df_res in results_all.items():
    print(f"## Résultats {task}")
    print(df_res.to_markdown(index=False))
    df = loader()
    rows=[]
    # for mid in MODELS:
    #     ref, minus_row, minus_col = find_critical_para(model, tok, device)
    #     best, auprc = eval_dataset(mid, model, tok, device, df, ref, minus_row, minus_col, task)
    #     rows.append({
    #         "model": mid.split("/")[-1],
    #         "precision": round(best['p'],3),
    #         "recall": round(best['r'],3),
    #         "f1": round(best['f1'],3),
    #         "auprc": round(auprc,3)
    #     })
    print(f"\n## Résultats {task}\n")
    print(pd.DataFrame(rows).to_markdown(index=False))