# scripts/predict_and_eval_distilbert.py
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_from_disk
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_DIR = "models/distilbert_finetuned"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

test_ds = load_from_disk("data/hf_test_dataset")
comments = test_ds["comment"]
labels = test_ds["label"]

device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)

pred_names = []
pred_ids = []

label_map = json.load(open("models/id2label.json"))
reverse_map = {v: k for k, v in label_map.items()}

print("Running predictions...")
for c in comments:
    scores = clf(c)[0]
    best = max(scores, key=lambda x: x["score"])
    pred_names.append(best["label"])
    pred_ids.append(int(reverse_map[best["label"]]))

# Save predictions
df = pd.DataFrame({
    "comment": comments,
    "true_label": labels,
    "pred_label_name": pred_names,
    "pred_label": pred_ids
})
df.to_csv("results/distilbert_test_predictions.csv", index=False)
print("Saved distilbert_test_predictions.csv")

# Classification report
report = classification_report(labels, pred_ids, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv("results/distilbert_classification_report.csv")
print("Saved distilbert_classification_report.csv")

# Confusion Matrix
cm = confusion_matrix(labels, pred_ids)
label_names = [label_map[str(i)] for i in sorted(label_map.keys(), key=int)]

pd.DataFrame(cm, index=label_names, columns=label_names).to_csv("results/distilbert_confusion_matrix.csv")

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=label_names, yticklabels=label_names)
plt.title("DistilBERT Confusion Matrix")
plt.savefig("results/distilbert_confusion_matrix.png", bbox_inches='tight')
plt.close()
print("Saved confusion matrix image")

# Misclassified examples
mis = df[df["true_label"] != df["pred_label"]]
mis.sample(100, random_state=42).to_csv("results/distilbert_misclassified_sample_100.csv", index=False)
print("Saved distilbert_misclassified_sample_100.csv")


# # scripts/predict_and_eval_distilbert.py = for predicting and evaluating DistilBERT model

# import json
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# from datasets import load_from_disk
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# MODEL_NAME = "distilbert"
# MODEL_DIR = f"models/{MODEL_NAME}_finetuned"

# print("🔹 Loading DistilBERT model and tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# print("🔹 Loading test dataset...")
# test_ds = load_from_disk("data/hf_test_dataset")
# comments = test_ds["comment"]
# true_labels = test_ds["label"]

# device = 0 if torch.cuda.is_available() else -1
# clf = pipeline(
#     "text-classification",
#     model=model,
#     tokenizer=tokenizer,
#     return_all_scores=True,
#     device=device
# )

# # Load label maps
# id2label = json.load(open("models/id2label.json"))
# label2id = {v: int(k) for k, v in id2label.items()}
# label_names = [id2label[str(i)] for i in range(len(id2label))]

# pred_names = []
# pred_ids = []
# prob_rows = []

# print("🔹 Running DistilBERT predictions...")
# for c in comments:
#     outputs = clf(c, truncation=True, max_length=128)[0]
#     best = max(outputs, key=lambda x: x["score"])

#     pred_names.append(best["label"])
#     pred_ids.append(label2id[best["label"]])

#     row = [0.0] * len(label_names)
#     for o in outputs:
#         row[label2id[o["label"]]] = o["score"]
#     prob_rows.append(row)

# # === Save probability CSV for ensemble ===
# pred_df = pd.DataFrame(prob_rows, columns=label_names)
# pred_df["true_label_id"] = true_labels
# pred_df["pred_label_id"] = pred_ids

# os.makedirs("predictions", exist_ok=True)
# pred_df.to_csv(f"predictions/{MODEL_NAME}_test_predictions.csv", index=False)
# print(f"✅ Saved softmax predictions to: predictions/{MODEL_NAME}_test_predictions.csv")

# # === Save regular outputs ===
# df = pd.DataFrame({
#     "comment": comments,
#     "true_label": true_labels,
#     "pred_label_name": pred_names,
#     "pred_label": pred_ids
# })

# os.makedirs("results", exist_ok=True)
# df.to_csv(f"results/{MODEL_NAME}_test_predictions.csv", index=False)
# print(f"✅ Saved: results/{MODEL_NAME}_test_predictions.csv")

# # Classification report
# report = classification_report(true_labels, pred_ids, output_dict=True, zero_division=0)
# pd.DataFrame(report).transpose().to_csv(f"results/{MODEL_NAME}_classification_report.csv")
# print(f"✅ Saved: results/{MODEL_NAME}_classification_report.csv")

# # Confusion matrix
# cm = confusion_matrix(true_labels, pred_ids)
# pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(f"results/{MODEL_NAME}_confusion_matrix.csv")

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=label_names, yticklabels=label_names)
# plt.title("DistilBERT Confusion Matrix")
# plt.savefig(f"results/{MODEL_NAME}_confusion_matrix.png", bbox_inches="tight")
# plt.close()
# print(f"✅ Saved: results/{MODEL_NAME}_confusion_matrix.png")

# # Misclassified samples
# mis = df[df["true_label"] != df["pred_label"]]
# sample_size = min(100, len(mis))
# mis.sample(sample_size, random_state=42).to_csv(f"results/{MODEL_NAME}_misclassified_sample_100.csv", index=False)
# print(f"✅ Saved: results/{MODEL_NAME}_misclassified_sample_100.csv")

# print("✅ DistilBERT evaluation complete!")
