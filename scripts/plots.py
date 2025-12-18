import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# 1) Load model comparison file
cmp = pd.read_csv("results/model_comparison.csv")

plt.figure(figsize=(8,5))
sns.barplot(x='model', y='accuracy', data=cmp)
plt.ylim(0,1)
plt.title("Model accuracy comparison")
plt.savefig("results/model_accuracy_comparison.png", bbox_inches='tight')
plt.close()

# 2) Load confusion matrix of best model
best_model = cmp.sort_values("accuracy", ascending=False).iloc[0]['model']

cm = pd.read_csv(f"results/{best_model}_confusion_matrix.csv", index_col=0)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion matrix - {best_model}")
plt.savefig(f"results/{best_model}_confusion_matrix.png", bbox_inches='tight')
plt.close()

print("Plots saved in results/")
