import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/model_f1_comparison_table.csv")

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("tight")
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.1, 1.4)

# Bold header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")

plt.title(
    "Macro F1-score Comparison Across Models",
    fontsize=14,
    pad=12
)

plt.savefig("results/model_f1_comparison_table.png", dpi=300, bbox_inches="tight")
plt.savefig("results/model_f1_comparison_table.pdf", bbox_inches="tight")
plt.show()
