from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to base model and your trained adapter
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "./my-model/final"
output_path = "./merged-model"

# Load base model and merge adapter
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_path)

print("Merge complete!")
