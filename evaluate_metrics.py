from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./lora-output").cuda()
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

# Load evaluation set (sample)
dataset = load_dataset("Adun/thaifood-cot", split="train[:100]")

rouge = load_metric("rouge")
bleu = load_metric("bleu")

results = []
for example in dataset:
    prompt = f"Question: {example['Question']}\nThink: {example['Think']}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    ref = example["Answer"]
    results.append({"pred": pred, "ref": ref})

# Format for metrics
preds = [r["pred"] for r in results]
refs = [[r["ref"].split()] for r in results]

print("\n🔹 ROUGE:")
print(rouge.compute(predictions=preds, references=[r["ref"] for r in results], use_stemmer=True))

print("\n🔹 BLEU:")
print(bleu.compute(predictions=[p.split() for p in preds], references=refs))