import os
import json
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate

# ------------------------------
# ✅ Load data and label mappings
# ------------------------------
df = pd.read_csv("data/prepared_data.csv")
with open("data/label_mappings.json") as f:
    mappings = json.load(f)

label2id = {k: int(v) for k, v in mappings["label2id"].items()}
id2label = {int(k): v for k, v in mappings["id2label"].items()}

# ------------------------------
# ✅ Train/val split
# ------------------------------
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# ------------------------------
# ✅ Tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["comment"], truncation=True, padding=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

# Drop unused index column if present
for ds in [train_dataset, val_dataset]:
    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])

# ------------------------------
# ✅ Load model & freeze layers
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

for param in model.bert.encoder.layer[:8].parameters():
    param.requires_grad = False
print("🔒 Frozen encoder layers 0–7")

# ------------------------------
# ✅ Metrics
# ------------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
    }

# ------------------------------
# ✅ Phase 1 — train head
# ------------------------------
args1 = TrainingArguments(
    output_dir="results/freeze_phase",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_steps=100,
    save_total_limit=2,
    logging_dir="results/freeze_phase/logs",
    logging_steps=20,
    seed=42
)

trainer1 = Trainer(
    model=model,
    args=args1,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer1.train()
metrics1 = trainer1.evaluate()
print("📊 Evaluation after Phase 1 (head only):")
print(metrics1)

# 🔓 Unfreeze all layers
for param in model.bert.encoder.parameters():
    param.requires_grad = True
print("🔓 Unfroze all encoder layers")

# ------------------------------
# ✅ Phase 2 — full fine-tuning
# ------------------------------
args2 = TrainingArguments(
    output_dir="results/unfreeze_phase",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_steps=100,
    save_total_limit=2,
    logging_dir="results/unfreeze_phase/logs",
    logging_steps=20,
    seed=42
)

trainer2 = Trainer(
    model=model,
    args=args2,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer2.train()
metrics2 = trainer2.evaluate()
print("✅ Final Evaluation after full fine-tuning:")
print(metrics2)

# Save final metrics
os.makedirs("results", exist_ok=True)
pd.DataFrame([metrics2]).to_csv("results/final_metrics_freeze_unfreeze.csv", index=False)
