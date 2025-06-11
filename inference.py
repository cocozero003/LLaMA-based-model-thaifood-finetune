from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("./merged-fp16-model").cuda()
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
if __name__ == "__main__":
    prompt = "Question: แกงส้มมีประโยชน์อย่างไร?
Think: สมุนไพรหลักมีอะไรบ้าง
Answer:"
    print(chat(prompt))