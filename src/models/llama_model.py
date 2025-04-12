import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id: str, device: str = 'cuda'):
    """
    Loads the Llama (or other) model from model_id and returns (model, tokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer