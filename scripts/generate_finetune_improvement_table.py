import matplotlib.pyplot as plt

# Table data
columns = [
    "Model",
    "Base Macro F1",
    "Fine-Tuned Macro F1",
    "Improvement"
]

data = [
    ["BERT", "0.8090", "0.8133", "+0.43%"],
    ["RoBERTa", "0.7920", "0.8065", "+1.45%"],
    ["ALBERT", "0.7790", "0.8028", "+2.38%"],
    ["DistilBERT", "0.7930", "0.8006", "+0.76%"],
]

# Create figure
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

# Create table
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc="center",
    cellLoc="center"
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Bold header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_linewidth(1.5)
    else:
        cell.set_linewidth(1.0)

# Title
plt.title(
    "Base Models vs Fine-Tuning Models : Fine-Tuning Improvement Analysis",
    fontsize=16,
    fontweight="bold",
    pad=20
)

# Save image
plt.savefig(
    "finetuning_improvement_table.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
