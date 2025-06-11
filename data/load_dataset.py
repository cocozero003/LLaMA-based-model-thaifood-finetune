from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(model_id="deepseek-ai/deepseek-llm-7b-base", split="train[:1000]"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("Adun/thaifood-cot", split=split)

    def format(example):
        prompt = f"Question: {example['Question']}\nThink: {example['Think']}\nAnswer:"
        target = example["Answer"]
        return tokenizer(prompt + target, truncation=True, padding="max_length", max_length=512)

    return dataset.map(format)