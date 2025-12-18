# scripts/train_distilbert.py

# -----------------------------
# ABSOLUTE FIX FOR MAC MPS BUG
# -----------------------------
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

import os, json, numpy as np, pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.utils.class_weight import compute_class_weight
import evaluate

device = torch.device("cpu")
print("Forced device:", device)

# -------------------------
# Load Datasets
# -------------------------
train_ds = load_from_disk("data/hf_train_dataset")
test_ds = load_from_disk("data/hf_test_dataset")

print("Train examples:", len(train_ds), " Test examples:", len(test_ds))

label2id = json.load(open("models/label2id.json"))
id2label = json.load(open("models/id2label.json"))
print("Labels:", label2id)

# -------------------------
# Load DistilBERT + Tokenizer
# -------------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

# -------------------------
# Tokenization
# -------------------------
def preprocess(ex):
    return tokenizer(ex["comment"], truncation=True, max_length=128)

print("Tokenizing...")
train_ds = train_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

# -------------------------
# Class Weights
# -------------------------
y_train = np.array(train_ds["label"])
w = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
w = torch.tensor(w, dtype=torch.float32).to(device)
print("Class weights:", w)

# -------------------------
# Custom Trainer (compatible with your transformers)
# -------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"].to(device)

        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        outputs = model(**batch)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# -------------------------
# Metrics
# -------------------------
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    return {
        "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# -------------------------
# TrainingArguments — SAFE for your transformers version
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    save_strategy="no",      # <-- IMPORTANT FIX
    logging_steps=50,
    weight_decay=0.01,
    fp16=False,
    seed=42,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------
# Train
# -------------------------
print("Training DistilBERT...")
trainer.train()

# -------------------------
# Save Model
# -------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -------------------------
# Evaluate
# -------------------------
metrics = trainer.evaluate()
print(metrics)
pd.DataFrame([metrics]).to_csv("results/distilbert_final_metrics.csv", index=False)
