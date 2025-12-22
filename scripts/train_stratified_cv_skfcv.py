# scripts/train_stratified_cv_skfcv.py
import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import evaluate

# -----------------------------
# Mac M4 Compatibility Fix (no MPS)
# -----------------------------
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# -----------------------------
# Load label mappings
# -----------------------------
with open("data/label_mappings.json") as f:
    label_data = json.load(f)
label2id = label_data["label2id"]
id2label = {int(k): v for k, v in label_data["id2label"].items()}

# -----------------------------
# Load and preprocess data
# -----------------------------
df = pd.read_csv("data/cleaned_comments.csv")
df = df.rename(columns={"emotion": "label"})
df["label"] = df["label"].map(label2id)

# -----------------------------
# Tokenizer and evaluation metrics
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def tokenize_function(example):
    return tokenizer(example["comment"], truncation=True, padding=True, max_length=128)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
    }

# -----------------------------
# 5-Fold Stratified CV Training
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df["comment"], df["label"])):
    print(f"\n🚀 Fold {fold+1}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    output_dir = f"models/fold_{fold+1}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_eval=True,
        save_steps=100,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        fp16=False,  # Set to True if using GPU (not on CPU)
        load_best_model_at_end=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)

    metrics = trainer.evaluate()
    pd.DataFrame([metrics]).to_csv(f"results/fold_{fold+1}_metrics.csv", index=False)

print("\n✅ Stratified CV training complete.")
