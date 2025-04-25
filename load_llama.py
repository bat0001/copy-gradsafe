from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "./model/Llama-3.2-3B-Instruct"

# Télécharger et sauvegarder le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)

# Télécharger et sauvegarder le modèle
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
model.save_pretrained(output_dir)

print(f"✅ Modèle complet sauvegardé dans : {output_dir}")
