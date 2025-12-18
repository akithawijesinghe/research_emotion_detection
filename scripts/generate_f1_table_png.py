# scripts/generate_f1_table_png.py

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# Final model results
# =========================
data = {
    "Model": ["BERT", "DistilBERT", "RoBERTa", "ALBERT"],
    "Anger": [0.682, 0.699, 0.688, 0.683],
    "Anxiety": [0.774, 0.740, 0.761, 0.729],
    "Fear": [0.811, 0.809, 0.811, 0.772],
    "Grief": [0.786, 0.798, 0.794, 0.793],
    "Sadness": [0.752, 0.754, 0.760, 0.706],
    "Avg": [0.761, 0.760, 0.763, 0.736],
}

df = pd.DataFrame(data)

# =========================
# Create figure
# =========================
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")

# =========================
# Table content
# =========================
table = ax.table(
    cellText=df.round(3).values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center",
)

# =========================
# Styling (paper-like)
# =========================
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

# Remove all borders first
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_linewidth(0)

# Bold header
for col in range(len(df.columns)):
    header = table[(0, col)]
    header.set_text_props(weight="bold")

# Draw booktabs-style rules
n_cols = len(df.columns)

# Top rule
for col in range(n_cols):
    cell = table[(0, col)]
    cell.visible_edges = "T"
    cell.set_edgecolor("black")
    cell.set_linewidth(1.2)

# Mid rule (below header)
for col in range(n_cols):
    cell = table[(1, col)]
    cell.visible_edges = "T"
    cell.set_edgecolor("black")
    cell.set_linewidth(0.8)

# Bottom rule
last_row = len(df)
for col in range(n_cols):
    cell = table[(last_row, col)]
    cell.visible_edges = "B"
    cell.set_edgecolor("black")
    cell.set_linewidth(1.2)

# =========================
# Title (paper style)
# =========================
plt.title(
    "Macro F1-score Comparison Across Transformer Models",
    fontsize=13,
    pad=10,
)

# =========================
# Save PNG
# =========================
plt.savefig(
    "results/macro_f1_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Saved PNG to results/macro_f1_comparison.png")
