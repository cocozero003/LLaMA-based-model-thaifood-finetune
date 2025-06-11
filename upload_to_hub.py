from huggingface_hub import notebook_login
from transformers import AutoModelForCausalLM, AutoTokenizer

notebook_login()

model = AutoModelForCausalLM.from_pretrained("./merged-fp16-model")
tokenizer = AutoTokenizer.from_pretrained("./merged-fp16-model")

model.push_to_hub("YourUsername/deepseek-llm-thaifood-cot")
tokenizer.push_to_hub("YourUsername/deepseek-llm-thaifood-cot")