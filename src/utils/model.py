from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_id, device="cuda:0"):
    print(f"Loading {model_id} …")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    return model, tok, next(model.parameters()).device