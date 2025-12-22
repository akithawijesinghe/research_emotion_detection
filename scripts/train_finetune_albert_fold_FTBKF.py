import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

# =========================
# 🔁 MODEL CONFIG
# =========================
MODEL_NAME = "albert-base-v2"
OUTPUT_DIR = "models/albert_finetune"   # ✅ CHANGED

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("data/prepared_data.csv")

with open("data/label_mappings.json") as f:
    label_data = json.load(f)

label2id = label_data["label2id"]
id2label = {int(k): v for k, v in label_data["id2label"].items()}
df["label"] = df["label"].astype(int)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# -------------------------
# Tokenization
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["comment"],
        truncation=True,
        padding=True,
        max_length=128
    )

train_dataset = Dataset.from_pandas(
    train_df[["comment", "label"]]
).map(tokenize, batched=True)

val_dataset = Dataset.from_pandas(
    val_df[["comment", "label"]]
).map(tokenize, batched=True)

# -------------------------
# Load model (NO freezing)
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id={v: int(k) for k, v in id2label.items()}
)

# -------------------------
# Metrics
# -------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(
            predictions=preds,
            references=p.label_ids
        )["accuracy"],
        "f1_macro": f1.compute(
            predictions=preds,
            references=p.label_ids,
            average="macro"
        )["f1"]
    }

# -------------------------
# Training Arguments
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    do_eval=True,
    save_total_limit=2,
    report_to="none",
    seed=42
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# -------------------------
# Train & Evaluate
# -------------------------
trainer.train()
metrics = trainer.evaluate()
print("✅ ALBERT FINETUNE METRICS:", metrics)
