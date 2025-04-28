from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_id: str):
    """FP16 + gradient‑checkpointing; retourne model, tokenizer, device."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer, next(model.parameters()).device