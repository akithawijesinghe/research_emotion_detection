# scripts/predict_and_eval.py
"""
Run inference with the fine-tuned RoBERTa model, save predictions, produce classification report,
confusion matrix and extract 100 misclassified samples for manual error analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_DIR = "models/roberta_finetuned"
TEST_DS_DIR = "data/hf_test_dataset"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Load model + tokenizer
print("Loading model and tokenizer from", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Load test dataset (huggingface dataset saved to disk)
print("Loading test dataset from", TEST_DS_DIR)
test_ds = load_from_disk(TEST_DS_DIR)

# Ensure test_ds has "comment" and "label"
assert "comment" in test_ds.column_names and "label" in test_ds.column_names

# Tokenize test dataset (same preprocessing as training)
def preprocess(batch):
    return tokenizer(batch["comment"], truncation=True, max_length=128)

print("Tokenizing test dataset...")
test_ds = test_ds.map(preprocess, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create a Trainer (for predict convenience)
trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer)

# Run predictions (batched)
print("Running predictions on test set...")
pred_output = trainer.predict(test_ds)   # returns PredictionOutput(predictions, label_ids, metrics)
logits = pred_output.predictions
if isinstance(logits, tuple):  # sometimes returns (logits, hidden_states)
    logits = logits[0]
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
pred_ids = np.argmax(probs, axis=1)
true_ids = pred_output.label_ids

# Load id2label map (strings)
id2label = json.load(open(os.path.join("models", "id2label.json")))
id2label = {int(k): v for k, v in id2label.items()}

pred_names = [id2label[int(i)] for i in pred_ids]
true_names = [id2label[int(i)] for i in true_ids]

# Build DataFrame with probs per class
class_cols = [id2label[i] for i in sorted(id2label.keys())]
probs_df = pd.DataFrame(probs, columns=class_cols)

df = pd.DataFrame({
    "comment": test_ds["comment"],
    "true_label_id": true_ids,
    "true_label": true_names,
    "pred_label_id": pred_ids,
    "pred_label": pred_names,
})
df = pd.concat([df, probs_df.reset_index(drop=True)], axis=1)

# Save full predictions
pred_csv = os.path.join(OUT_DIR, "roberta_test_predictions.csv")
df.to_csv(pred_csv, index=False)
print("Saved predictions to", pred_csv)

# Classification report (sklearn expects integer labels)
labels_sorted = sorted(list(id2label.keys()))
target_names = [id2label[i] for i in labels_sorted]
report = classification_report(true_ids, pred_ids, labels=labels_sorted, target_names=target_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_csv = os.path.join(OUT_DIR, "roberta_classification_report.csv")
report_df.to_csv(report_csv)
print("Saved classification report to", report_csv)

# Confusion matrix
cm = confusion_matrix(true_ids, pred_ids, labels=labels_sorted)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
cm_csv = os.path.join(OUT_DIR, "roberta_confusion_matrix.csv")
cm_df.to_csv(cm_csv)
print("Saved confusion matrix CSV to", cm_csv)

# Save confusion matrix plot
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("RoBERTa Confusion Matrix (Test set)")
plt.tight_layout()
cm_png = os.path.join(OUT_DIR, "roberta_confusion_matrix.png")
plt.savefig(cm_png, bbox_inches="tight")
plt.close()
print("Saved confusion matrix image to", cm_png)

# Extract misclassified samples
mis_df = df[df["true_label_id"] != df["pred_label_id"]].copy()
print("Total misclassified examples:", len(mis_df))
if len(mis_df) == 0:
    print("No misclassified examples found (unexpected).")
else:
    sample_count = min(100, len(mis_df))
    mis_sample = mis_df.sample(sample_count, random_state=42).reset_index(drop=True)
    mis_sample_csv = os.path.join(OUT_DIR, "misclassified_sample_100_tuned.csv")
    mis_sample.to_csv(mis_sample_csv, index=False)
    print(f"Saved {sample_count} misclassified samples to", mis_sample_csv)

print("All done.")





# # scripts/predict_and_eval_roberta.py

# import os
# import json
# import numpy as np
# import pandas as pd
# from datasets import load_from_disk
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
# import torch
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# MODEL_NAME = "roberta"
# MODEL_DIR = f"models/{MODEL_NAME}_finetuned"
# TEST_DS_DIR = "data/hf_test_dataset"
# OUT_DIR = "results"
# PRED_DIR = "predictions"
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(PRED_DIR, exist_ok=True)

# # === 1. Load model and tokenizer ===
# print(f"🔹 Loading {MODEL_NAME} model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# # === 2. Load and preprocess test dataset ===
# print("🔹 Loading test dataset...")
# test_ds = load_from_disk(TEST_DS_DIR)

# assert "comment" in test_ds.column_names and "label" in test_ds.column_names

# def preprocess(batch):
#     return tokenizer(batch["comment"], truncation=True, max_length=128)

# test_ds = test_ds.map(preprocess, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# trainer = Trainer(model=model, data_collator=data_collator, tokenizer=tokenizer)

# # === 3. Run inference ===
# print("🔹 Running predictions...")
# pred_output = trainer.predict(test_ds)
# logits = pred_output.predictions
# if isinstance(logits, tuple):
#     logits = logits[0]
# probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
# pred_ids = np.argmax(probs, axis=1)
# true_ids = pred_output.label_ids

# # === 4. Load label maps ===
# id2label = json.load(open("models/id2label.json"))
# id2label = {int(k): v for k, v in id2label.items()}
# label_names = [id2label[i] for i in sorted(id2label.keys())]

# pred_names = [id2label[i] for i in pred_ids]
# true_names = [id2label[i] for i in true_ids]

# # === 5. Save prediction CSV with softmax for ensemble ===
# probs_df = pd.DataFrame(probs, columns=label_names)
# probs_df["true_label_id"] = true_ids
# probs_df["pred_label_id"] = pred_ids

# probs_df.to_csv(f"{PRED_DIR}/{MODEL_NAME}_test_predictions.csv", index=False)
# print(f"✅ Saved softmax predictions: {PRED_DIR}/{MODEL_NAME}_test_predictions.csv")

# # === 6. Save regular prediction output ===
# df = pd.DataFrame({
#     "comment": test_ds["comment"],
#     "true_label_id": true_ids,
#     "true_label": true_names,
#     "pred_label_id": pred_ids,
#     "pred_label": pred_names,
# })
# df = pd.concat([df.reset_index(drop=True), probs_df[label_names].reset_index(drop=True)], axis=1)
# df.to_csv(f"{OUT_DIR}/{MODEL_NAME}_test_predictions.csv", index=False)
# print(f"✅ Saved: {OUT_DIR}/{MODEL_NAME}_test_predictions.csv")

# # === 7. Classification report ===
# report = classification_report(true_ids, pred_ids, target_names=label_names, output_dict=True, zero_division=0)
# pd.DataFrame(report).transpose().to_csv(f"{OUT_DIR}/{MODEL_NAME}_classification_report.csv")
# print(f"✅ Saved: {OUT_DIR}/{MODEL_NAME}_classification_report.csv")

# # === 8. Confusion matrix ===
# cm = confusion_matrix(true_ids, pred_ids)
# cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
# cm_df.to_csv(f"{OUT_DIR}/{MODEL_NAME}_confusion_matrix.csv")
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
# plt.title(f"{MODEL_NAME.upper()} Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/{MODEL_NAME}_confusion_matrix.png")
# plt.close()
# print(f"✅ Saved: {OUT_DIR}/{MODEL_NAME}_confusion_matrix.png")

# # === 9. Misclassified samples ===
# mis = df[df["true_label_id"] != df["pred_label_id"]]
# sample_count = min(100, len(mis))
# mis.sample(sample_count, random_state=42).to_csv(f"{OUT_DIR}/{MODEL_NAME}_misclassified_sample_100.csv", index=False)
# print(f"✅ Saved: {OUT_DIR}/{MODEL_NAME}_misclassified_sample_100.csv")

# print("✅ RoBERTa evaluation complete.")
