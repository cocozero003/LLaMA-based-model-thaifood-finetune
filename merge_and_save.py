from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base", device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, "./lora-output")

peft_model = peft_model.merge_and_unload()
peft_model.save_pretrained("./merged-fp16-model")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
tokenizer.save_pretrained("./merged-fp16-model")

print("✅ Model merged and saved to ./merged-fp16-model")