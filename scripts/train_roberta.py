# scripts/train_roberta.py
"""
Fine-tune roberta-base for multi-class emotion classification.
This version is robust to Trainer API differences (accepts extra kwargs in compute_loss).
Outputs:
 - model saved to models/roberta_finetuned
 - results/roberta_final_metrics.csv
"""

import os
import json
import numpy as np
from datasets import load_from_disk
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "roberta-base"                 # change to "distilbert-base-uncased" if CPU too slow
OUTPUT_DIR = "models/roberta_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

# Hyperparams (tune if needed)
BATCH_SIZE = 8      # use 8 on GPU; use 4 (or 2) on CPU/MPS
EPOCHS = 3
LR = 2e-5
SEED = 42
MAX_LENGTH = 128

# ----------------------------
# Load datasets saved earlier
# ----------------------------
print("Loading HF datasets from disk...")
train_ds = load_from_disk("data/hf_train_dataset")
test_ds = load_from_disk("data/hf_test_dataset")
print("Train examples:", len(train_ds), "Test examples:", len(test_ds))

# ----------------------------
# Load label maps
# ----------------------------
label2id = json.load(open("models/label2id.json"))
id2label = json.load(open("models/id2label.json"))
NUM_LABELS = len(label2id)
print("Labels (label2id):", label2id)

# ----------------------------
# Tokenizer and model
# ----------------------------
print("Loading tokenizer and model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(batch):
    return tokenizer(batch["comment"], truncation=True, max_length=MAX_LENGTH)

print("Tokenizing datasets...")
train_ds = train_ds.map(preprocess, batched=True)
test_ds = test_ds.map(preprocess, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------
# Compute class weights
# ----------------------------
y_train = np.array(train_ds["label"])
class_weights_np = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print("Class weights (numpy):", class_weights_np)
print("Class weights (tensor):", class_weights)

# ----------------------------
# Custom Trainer with weighted loss (robust signature)
# ----------------------------
from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute weighted cross-entropy loss.
        Accepts extra kwargs for compatibility with different Trainer versions
        (e.g., num_items_in_batch).
        """
        # Move class_weights to the model device (CPU/MPS/CUDA)
        device = next(model.parameters()).device
        cw = class_weights.to(device)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# ----------------------------
# Metrics
# ----------------------------
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")
    prec = precision_metric.compute(predictions=preds, references=labels, average="macro")
    rec = recall_metric.compute(predictions=preds, references=labels, average="macro")
    # Ensure scalar values are returned (not nested dicts)
    return {
        "accuracy": float(acc["accuracy"]),
        "f1_macro": float(f1_macro["f1"]),
        "precision_macro": float(prec["precision"]),
        "recall_macro": float(rec["recall"]),
    }

# ----------------------------
# TrainingArguments (minimal & compatible)
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    seed=SEED,
    fp16=False,  # set True only if GPU with fp16 supported
    # Note: avoid evaluation_strategy/save_strategy/load_best_model_at_end to stay compatible with user's transformers
)

# ----------------------------
# Trainer init
# ----------------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ----------------------------
# Train
# ----------------------------
print("Starting training...")
trainer.train()
print("Training finished. Saving model and tokenizer...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ----------------------------
# Evaluate explicitly after training
# ----------------------------
print("Evaluating on test set (explicit call)...")
metrics = trainer.evaluate(eval_dataset=test_ds)
print("Evaluation metrics:", metrics)

# Save metrics to CSV
metrics_out = {
    "accuracy": float(metrics.get("eval_accuracy", metrics.get("accuracy", 0.0))),
    "f1_macro": float(metrics.get("eval_f1_macro", metrics.get("f1_macro", 0.0))),
    "precision_macro": float(metrics.get("eval_precision_macro", metrics.get("precision_macro", 0.0))),
    "recall_macro": float(metrics.get("eval_recall_macro", metrics.get("recall_macro", 0.0))),
}
pd.DataFrame([metrics_out]).to_csv("results/roberta_final_metrics.csv", index=False)
print("Saved results/roberta_final_metrics.csv")

print("Done.")
