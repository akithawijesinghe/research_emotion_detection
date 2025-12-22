# scripts/predict_and_eval_bert_finetune_FTBKF.py

import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Load model from last checkpoint
# ------------------------
print("🔹 Loading BERT model and tokenizer...")

MODEL_PATH = "results/bert_finetune_fold0/checkpoint-3822"  # ✅ Change if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# ------------------------
# Load test data
# ------------------------
print("🔹 Loading test dataset...")
test_df = pd.read_csv("data/test_data.csv")

with open("data/label_mappings.json") as f:
    label_data = json.load(f)
id2label = {int(k): v for k, v in label_data["id2label"].items()}
label2id = label_data["label2id"]

test_df["label"] = test_df["label"].astype(int)
test_dataset = Dataset.from_pandas(test_df[["comment", "label"]])

# ------------------------
# Run predictions
# ------------------------
print("🔹 Running BERT predictions...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
    truncation=True
)

preds = []
probs = []

for text in test_df["comment"]:
    out = pipe(text)[0]  # Now returns list of dicts (one per class)
    scores = [x["score"] for x in out]
    label_ids = [label2id[x["label"]] for x in out]
    pred_label = label_ids[np.argmax(scores)]
    preds.append(pred_label)
    probs.append(scores)

# ------------------------
# Save outputs
# ------------------------
os.makedirs("results", exist_ok=True)

# Softmax predictions
prob_df = pd.DataFrame(probs, columns=[f"prob_{id2label[i]}" for i in range(len(id2label))])
prob_df["true_label"] = test_df["label"]
prob_df["pred_label"] = preds
prob_df.to_csv("results/bert_finetune_test_predictions.csv", index=False)
print("✅ Saved: results/bert_finetune_test_predictions.csv")

# Classification Report
report = classification_report(test_df["label"], preds, target_names=list(label2id.keys()), output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("results/bert_finetune_classification_report.csv")
print("✅ Saved: results/bert_finetune_classification_report.csv")

# Confusion Matrix
cm = confusion_matrix(test_df["label"], preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=id2label.values(), yticklabels=id2label.values())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - BERT Fine-tuned")
plt.tight_layout()
plt.savefig("results/bert_finetune_confusion_matrix.png")
print("✅ Saved: results/bert_finetune_confusion_matrix.png")

# Save misclassified examples
test_df["pred_label"] = preds
misclassified = test_df[test_df["label"] != test_df["pred_label"]]
misclassified.to_csv("results/bert_finetune_misclassified_sample_100.csv", index=False)
print("✅ Saved: results/bert_finetune_misclassified_sample_100.csv")

print("✅ BERT fine-tuned evaluation complete!")
