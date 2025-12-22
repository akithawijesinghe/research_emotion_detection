import os
import json
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import Dataset

# -------------------------
# ✅ Setup
# -------------------------
print("🔹 Loading BERT model and tokenizer...")

model_path = "results/unfreeze_phase"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device set to use {device}")

# -------------------------
# ✅ Load test data
# -------------------------
df = pd.read_csv("data/prepared_data.csv")
with open("data/label_mappings.json") as f:
    mappings = json.load(f)

label2id = mappings["label2id"]
id2label = {int(k): v for k, v in mappings["id2label"].items()}

from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
test_texts = test_df["comment"].tolist()
true_labels = test_df["label"].tolist()

# -------------------------
# ✅ Create pipeline
# -------------------------
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True
)

print("🔹 Running BERT predictions...")

# -------------------------
# ✅ Predict
# -------------------------
predictions = pipe(test_texts, batch_size=16)

# Convert softmax scores
softmax_scores = [pred for pred in predictions]
probs = [ [p['score'] for p in pred] for pred in softmax_scores]
predicted_ids = np.argmax(probs, axis=1)
predicted_labels = [id2label[i] for i in predicted_ids]
true_labels_str = [id2label[i] for i in true_labels]

# Save predictions
os.makedirs("predictions", exist_ok=True)
os.makedirs("results", exist_ok=True)

pd.DataFrame(probs, columns=[f"prob_{id2label[i]}" for i in range(len(id2label))])\
    .to_csv("predictions/bert_freeze_softmax_predictions.csv", index=False)

df_out = pd.DataFrame({
    "comment": test_texts,
    "true_label": true_labels_str,
    "predicted_label": predicted_labels
})
df_out.to_csv("results/bert_freeze_predictions.csv", index=False)

# -------------------------
# ✅ Classification report
# -------------------------
report = classification_report(true_labels_str, predicted_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("results/bert_freeze_classification_report.csv")
print("✅ Saved: results/bert_freeze_classification_report.csv")

# -------------------------
# ✅ Confusion matrix
# -------------------------
cm = confusion_matrix(true_labels_str, predicted_labels, labels=sorted(set(true_labels_str)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=sorted(set(true_labels_str)), yticklabels=sorted(set(true_labels_str)))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — BERT Freeze/Unfreeze")
plt.tight_layout()
plt.savefig("results/bert_freeze_confusion_matrix.png")
print("✅ Saved: results/bert_freeze_confusion_matrix.png")

# -------------------------
# ✅ Misclassified examples
# -------------------------
misclassified = df_out[df_out["true_label"] != df_out["predicted_label"]]
misclassified.head(100).to_csv("results/bert_freeze_misclassified_sample_100.csv", index=False)
print("✅ Saved: results/bert_freeze_misclassified_sample_100.csv")

print("✅ BERT Freeze-Unfreeze evaluation complete!")
