from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("Adun/thaifood-cot", split="train[:1000]")

def format(example):
    prompt = f"Question: {example['Question']}\nThink: {example['Think']}\nAnswer:"
    target = example["Answer"]
    return tokenizer(prompt + target, truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(format)

training_args = TrainingArguments(
    output_dir="./lora-output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()