import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

os.makedirs("results", exist_ok=True)

# Load baseline comparison file
cmp = pd.read_csv("results/model_comparison.csv")

# Try to load tuned classification report
try:
    tuned = pd.read_csv("results/logreg_tuned_classification_report.csv", index_col=0)

    # Get macro-F1
    macro_f1 = tuned.loc['macro avg', 'f1-score']

    # Load tuned confusion matrix to compute accuracy
    cm = pd.read_csv("results/logreg_tuned_confusion_matrix.csv", index_col=0)
    correct = cm.values.trace()
    total = cm.values.sum()
    accuracy = correct / total

    # 👉 FIX: Instead of append(), use pd.concat
    new_row = pd.DataFrame([{
        'model':'LogReg_Tuned',
        'accuracy':accuracy,
        'macro_f1':macro_f1
    }])

    cmp = pd.concat([cmp, new_row], ignore_index=True)

except Exception as e:
    print("Tuned metrics not found:", e)

# Plot accuracy comparison (updated)
plt.figure(figsize=(8,5))
sns.barplot(x='model', y='accuracy', data=cmp)
plt.ylim(0,1)
plt.title("Model accuracy comparison (including tuned LR)")
plt.xticks(rotation=15)
plt.savefig("results/model_accuracy_comparison_updated.png", bbox_inches='tight')
plt.close()

# Plot tuned confusion matrix if exists
if os.path.exists("results/logreg_tuned_confusion_matrix.csv"):
    cm2 = pd.read_csv("results/logreg_tuned_confusion_matrix.csv", index_col=0)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion matrix - LogReg Tuned")
    plt.savefig("results/logreg_tuned_confusion_matrix.png", bbox_inches='tight')
    plt.close()

print("Updated plots saved in results/")
