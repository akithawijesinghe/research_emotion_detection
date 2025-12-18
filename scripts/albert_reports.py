# scripts/albert_reports.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

df = pd.read_csv("results/albert_test_predictions.csv")
id2label = json.load(open("models/id2label.json"))

y_true = df["true_label"]
y_pred = df["pred_label"]

labels = [id2label[str(i)] for i in sorted(id2label.keys(), key=int)]

# Classification report
report = classification_report(
    y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
)
pd.DataFrame(report).transpose().to_csv("results/albert_classification_report.csv")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

pd.DataFrame(cm, index=labels, columns=labels).to_csv(
    "results/albert_confusion_matrix.csv"
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("ALBERT Confusion Matrix")
plt.savefig("results/albert_confusion_matrix.png", bbox_inches="tight")

print("Saved all ALBERT reports.")
