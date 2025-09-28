import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling

# 1. Load dataset
dataset = load_dataset("json", data_files="data.json")

# 2. Choose a base model (LLaMA-2 7B for example)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenize
def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# 4. Quantization setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit quantization
    bnb_4bit_use_double_quant=True, # second quantization for stability
    bnb_4bit_quant_type="nf4",      # NormalFloat4 (best for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 5. Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 6. LoRA config (added on top of quantized base)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 7. Training args
args = TrainingArguments(
    output_dir="./qlora_out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=200,  # demo run
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50
)

# 8. Data collator
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=collator
)

# 10. Train
trainer.train()

# Save LoRA adapters only
model.save_pretrained("./qlora_out")
