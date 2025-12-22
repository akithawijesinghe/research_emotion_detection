import matplotlib.pyplot as plt

# -----------------------------
# Table data
# -----------------------------
columns = [
    "Model",
    "Accuracy",
    "Macro F1",
    "Macro Precision",
    "Macro Recall"
]

data = [
    ["BERT (Base)", "0.8703", "0.8090", "0.8172", "0.8021"],
    ["DistilBERT (Base)", "0.7930", "0.7930", "0.8019", "0.7870"],
    ["RoBERTa (Base)", "0.7920", "0.7920", "0.7992", "0.7873"],
    ["ALBERT (Base)", "0.7790", "0.7790", "0.7927", "0.7722"],
    ["BERT Fine-Tuned", "0.8133", "0.8133", "0.8128", "0.8144"],
    ["RoBERTa Fine-Tuned", "0.8070", "0.8065", "0.8060", "0.8060"],
    ["ALBERT Fine-Tuned", "0.8027", "0.8028", "0.8030", "0.8030"],
    ["DistilBERT Fine-Tuned", "0.8007", "0.8006", "0.8010", "0.8010"],
]

# -----------------------------
# Create figure
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

# Title
plt.title(
    "Transformer Models: Performance Comparison",
    fontsize=20,
    fontweight="bold",
    pad=20
)

# Create table
table = ax.table(
    cellText=data,
    colLabels=columns,
    cellLoc="center",
    loc="center"
)

# -----------------------------
# Style table
# -----------------------------
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(1.5)

    # Header styling
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_height(0.08)

# -----------------------------
# Save image
# -----------------------------
plt.savefig(
    "final_transformer_performance_table.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("✅ PNG saved as: final_transformer_performance_table.png")
