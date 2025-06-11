# Meta LLaMA-2 Fine-Tuning: Thai Food CoT

This version fine-tunes `meta-llama/Llama-2-7b-hf` using the [ThaiFood CoT Dataset](https://huggingface.co/datasets/Adun/thaifood-cot) via LoRA.


- Uses Meta's official LLaMA-2 model
- Same LoRA PEFT strategy
- Drop-in replacement for DeepSeek

## Usage

### Install Requirements
```bash
pip install -r requirements.txt
```

### Train
```bash
python train.py
```