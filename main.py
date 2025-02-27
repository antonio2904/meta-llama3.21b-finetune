import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Load dataset
dataset = load_dataset("yelp_review_full")

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Configure BitsAndBytes quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # Enable 8-bit quantization for efficiency
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=5,
    device_map="auto"  # Auto-distribute across GPUs
)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))  # Ensure model recognizes new token

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)

# Print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} / {all_param} ({100 * trainable_params / all_param:.2f}%)")

print_trainable_parameters(model)

def preprocess_function(examples):
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    encoding["labels"] = [label - 1 for label in examples["label"]]  # Convert 1-5 → 0-4
    return encoding

# Select a subset of 1000 samples for training and evaluation
dataset["train"] = dataset["train"].select(range(2000))
dataset["test"] = dataset["test"].select(range(1000))

tokenized_datasets = dataset.map(preprocess_function, batched=True, batch_size=2)
small_train_dataset = tokenized_datasets["train"]
small_eval_dataset = tokenized_datasets["test"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
    fp16=True,
    deepspeed="ds_config.json"  # Use DeepSpeed
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    data_collator=data_collator
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./fine_tuned_llama3_yelp")
tokenizer.save_pretrained("./fine_tuned_llama3_yelp")

