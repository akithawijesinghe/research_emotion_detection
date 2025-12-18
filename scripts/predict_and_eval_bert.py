# scripts/predict_and_eval_bert.py
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_from_disk
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = "models/bert_finetuned"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

print("Loading test dataset...")
test_ds = load_from_disk("data/hf_test_dataset")
comments = test_ds["comment"]
labels = test_ds["label"]

device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, return_all_scores=True)

pred_names = []
pred_ids = []
prob_list = []

label_map = json.load(open("models/id2label.json"))
reverse_map = {v: k for k, v in label_map.items()}

print("Running predictions...")
for c in comments:
    outputs = clf(c, truncation=True, max_length=128)[0]
    best = max(outputs, key=lambda x: x['score'])
    pred_names.append(best['label'])
    pred_ids.append(int(reverse_map[best['label']]))
    prob_list.append({o['label']: o['score'] for o in outputs})

df = pd.DataFrame({
    "comment": comments,
    "true_label": labels,
    "pred_label": pred_ids,
    "pred_label_name": pred_names
})

df.to_csv("results/bert_test_predictions.csv", index=False)
print("Saved results/bert_test_predictions.csv")

# Classification report
report = classification_report(labels, pred_ids, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv("results/bert_classification_report.csv")
print("Saved bert_classification_report.csv")

# Confusion matrix
cm = confusion_matrix(labels, pred_ids)
label_names = [label_map[str(i)] for i in sorted(label_map.keys(), key=int)]

pd.DataFrame(cm, index=label_names, columns=label_names).to_csv("results/bert_confusion_matrix.csv")

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title("BERT Confusion Matrix")
plt.savefig("results/bert_confusion_matrix.png", bbox_inches='tight')
plt.close()
print("Saved bert_confusion_matrix.png")

# Misclassified 100 samples
mis = df[df["true_label"] != df["pred_label"]]
mis.sample(100, random_state=42).to_csv("results/bert_misclassified_sample_100.csv", index=False)
print("Saved bert_misclassified_sample_100.csv")
