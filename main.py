import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
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
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

print(f"PAD Token: {tokenizer.pad_token}, PAD Token ID: {tokenizer.pad_token_id}")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, torch_dtype=torch.float16)
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))  # Ensure model recognizes new token

def preprocess_function(examples):
    encoding = tokenizer(examples["text"], padding="longest", truncation=True, max_length=512)
    encoding["labels"] = [label - 1 for label in examples["label"]]  # Convert 1-5 → 0-4
    return encoding

tokenized_datasets = dataset.map(preprocess_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,
    fp16=True,
    bf16=False
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

