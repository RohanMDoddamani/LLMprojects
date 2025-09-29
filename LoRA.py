import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import DataCollatorForLanguageModeling, Trainer

# 1. Load dataset

newData = [
  {"instruction": "Who did Alice meet in the forest?", 
   "input": "", 
   "output": "Alice met a rabbit in the forest."},

  {"instruction": "Where did Alice go after meeting the rabbit?", 
   "input": "", 
   "output": "She went deeper into the forest."}
]

dataset = load_dataset("json", data_files="data.json")

# 2. Choose a base model (small for demo)
model_name = "meta-llama/Llama-2-7b-hf"  # you can also try "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenize
def tokenize_fn(example):
    text = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True)

# 4. Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,     # memory-efficient
    device_map="auto"
)

# 5. LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # common for LLaMA
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# 6. Training arguments
args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=200,  # small demo
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./lora_out",
    save_strategy="steps",
    save_steps=50
)

# 7. Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    data_collator=data_collator
)

# 8. Train
trainer.train()

# Save final LoRA adapter
model.save_pretrained("./lora_out")
