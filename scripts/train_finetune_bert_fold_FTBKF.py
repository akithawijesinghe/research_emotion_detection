import os
import json
import pandas as pd
import numpy as np
import torch
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

# ------------------------
# Load and Prepare Dataset
# ------------------------
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

# ------------------------
# Tokenization
# ------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["comment"], truncation=True, padding=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df[["comment", "label"]]).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df[["comment", "label"]]).map(tokenize, batched=True)

# ------------------------
# Load Model (No freezing)
# ------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id={v: int(k) for k, v in id2label.items()}
)

for param in model.bert.parameters():
    param.requires_grad = True

# ------------------------
# Metrics
# ------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
    }

# ------------------------
# TrainingArguments (SAFE version)
# ------------------------
training_args = TrainingArguments(
    output_dir="results/bert_finetune_fold0",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=200,
    logging_dir="logs",
    logging_steps=100,
    save_total_limit=2,
    do_eval=True,
    eval_steps=100,             # Eval every 100 steps
    save_steps=100,             # Save every 100 steps
    load_best_model_at_end=False,  # Disabled to avoid error in this version
    report_to="none",
    seed=42
)

# ------------------------
# Trainer
# ------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ------------------------
# Train and Evaluate
# ------------------------
trainer.train()
metrics = trainer.evaluate()
print("✅ Final evaluation metrics:", metrics)



# import os
# import json
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.model_selection import train_test_split
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     DataCollatorWithPadding
# )
# import evaluate

# # Load data and labels
# df = pd.read_csv("data/prepared_data.csv")
# with open("data/label_mappings.json") as f:
#     label_data = json.load(f)
# label2id = label_data["label2id"]
# id2label = {int(k): v for k, v in label_data["id2label"].items()}
# df["label"] = df["label"].astype(int)

# # Re-create the split and tokenization
# train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# def tokenize(batch): return tokenizer(batch["comment"], truncation=True, padding=True, max_length=128)
# train_dataset = Dataset.from_pandas(train_df[["comment", "label"]]).map(tokenize, batched=True)
# val_dataset = Dataset.from_pandas(val_df[["comment", "label"]]).map(tokenize, batched=True)

# # Load model in-place (just to rebind the object name `model`)
# model = AutoModelForSequenceClassification.from_pretrained(
#     "bert-base-uncased",
#     num_labels=len(label2id),
#     id2label=id2label,
#     label2id={v: int(k) for k, v in id2label.items()}
# )

# # Trainer re-instantiation (no training here)
# training_args = TrainingArguments(
#     output_dir="results/bert_finetune_fold0",
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=6,
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     save_total_limit=2,
#     logging_dir="logs",
#     save_steps=100,
#     eval_steps=100,
#     do_eval=True,
#     report_to="none"
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
#     data_collator=DataCollatorWithPadding(tokenizer)
# )
