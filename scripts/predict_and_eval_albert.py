# scripts/predict_and_eval_albert.py

import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = "models/albert_finetuned"

print("Loading ALBERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

print("Loading test dataset...")
test_ds = load_from_disk("data/hf_test_dataset")
comments = test_ds["comment"]
true_labels = test_ds["label"]

device = 0 if torch.cuda.is_available() else -1

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=device
)

# Load label maps
id2label = json.load(open("models/id2label.json"))
label2id = {v: int(k) for k, v in id2label.items()}

pred_names = []
pred_ids = []
prob_list = []

print("Running ALBERT predictions...")
for c in comments:
    outputs = clf(c, truncation=True, max_length=128)[0]
    best = max(outputs, key=lambda x: x["score"])

    pred_names.append(best["label"])
    pred_ids.append(label2id[best["label"]])
    prob_list.append({o["label"]: o["score"] for o in outputs})

# =========================
# SAVE PREDICTIONS
# =========================
df = pd.DataFrame({
    "comment": comments,
    "true_label": true_labels,
    "pred_label": pred_ids,
    "pred_label_name": pred_names
})

df.to_csv("results/albert_test_predictions.csv", index=False)
print("Saved results/albert_test_predictions.csv")

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(
    true_labels,
    pred_ids,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    "results/albert_classification_report.csv"
)
print("Saved albert_classification_report.csv")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(true_labels, pred_ids)

label_names = [id2label[str(i)] for i in sorted(id2label.keys(), key=int)]

pd.DataFrame(
    cm,
    index=label_names,
    columns=label_names
).to_csv("results/albert_confusion_matrix.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)
plt.title("ALBERT Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("results/albert_confusion_matrix.png", bbox_inches="tight")
plt.close()

print("Saved albert_confusion_matrix.png")

# =========================
# MISCLASSIFIED SAMPLES
# =========================
mis = df[df["true_label"] != df["pred_label"]]

sample_size = min(100, len(mis))
mis.sample(sample_size, random_state=42).to_csv(
    "results/albert_misclassified_sample_100.csv",
    index=False
)

print("Saved albert_misclassified_sample_100.csv")

print("✅ ALBERT evaluation complete!")
