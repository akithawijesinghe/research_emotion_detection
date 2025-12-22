import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 🔁 PATH CONFIG
# =========================
MODEL_DIR = "models/distilbert_finetune/checkpoint-3822"  # ✅ UPDATE IF NEEDED
OUTPUT_PREFIX = "distilbert_finetuned"

# -------------------------
# Load label mappings
# -------------------------
with open("data/label_mappings.json") as f:
    label_data = json.load(f)

label2id = label_data["label2id"]
id2label = {int(k): v for k, v in label_data["id2label"].items()}

# -------------------------
# Load model & tokenizer
# -------------------------
print("🔹 Loading DISTILBERT fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# Load test dataset
# -------------------------
print("🔹 Loading test dataset...")
test_df = pd.read_csv("data/test_data.csv")
test_df["label"] = test_df["label"].astype(int)

# -------------------------
# Prediction pipeline
# -------------------------
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
    truncation=True
)

preds = []
probs = []

print("🔹 Running DISTILBERT predictions...")
for text in test_df["comment"]:
    outputs = pipe(text)[0]
    scores = [o["score"] for o in outputs]
    preds.append(int(np.argmax(scores)))
    probs.append(scores)

# -------------------------
# Save results
# -------------------------
os.makedirs("results", exist_ok=True)

# ---- Probabilities
prob_df = pd.DataFrame(
    probs,
    columns=[f"prob_{id2label[i]}" for i in range(len(id2label))]
)
prob_df["true_label"] = test_df["label"]
prob_df["pred_label"] = preds

prob_df.to_csv(
    f"results/{OUTPUT_PREFIX}_test_predictions.csv",
    index=False
)
print("✅ Saved predictions CSV")

# ---- Classification report
report = classification_report(
    test_df["label"],
    preds,
    target_names=list(label2id.keys()),
    output_dict=True,
    zero_division=0
)
pd.DataFrame(report).transpose().to_csv(
    f"results/{OUTPUT_PREFIX}_classification_report.csv"
)
print("✅ Saved classification report")

# ---- Confusion matrix
cm = confusion_matrix(test_df["label"], preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=id2label.values(),
    yticklabels=id2label.values()
)
plt.title("Confusion Matrix – DISTILBERT Fine-tuned")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(
    f"results/{OUTPUT_PREFIX}_confusion_matrix.png"
)
plt.close()
print("✅ Saved confusion matrix")

# ---- Misclassified samples
mis_df = test_df.copy()
mis_df["pred_label"] = preds
mis_df = mis_df[mis_df["label"] != mis_df["pred_label"]]
mis_df.to_csv(
    f"results/{OUTPUT_PREFIX}_misclassified.csv",
    index=False
)
print("✅ Saved misclassified samples")

print("✅ DISTILBERT fine-tuned evaluation complete!")
