import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Ensure high-resolution text rendering
matplotlib.rcParams.update({'font.size': 14, 'font.weight': 'regular'})

# === 🔽 Step 1: Load CSV (Ensure this is in the same directory or update the path) ===
csv_path = "results/macro_f1_comparison_table.csv"  # adjust path as needed
df = pd.read_csv(csv_path)

# === 🔽 Step 2: Create Figure & Table ===
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")

# Format DataFrame for better presentation
table_data = [df.columns.to_list()] + df.values.tolist()

# === 🔽 Step 3: Draw Table ===
table = ax.table(
    cellText=table_data,
    colLabels=None,
    cellLoc="center",
    loc="center",
    edges="horizontal",
)

# Style adjustments
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1.2, 1.5)

# Bold header row
for key, cell in table.get_celld().items():
    if key[0] == 0:  # header row
        cell.set_text_props(weight="bold")

# Add title
plt.title("Macro F1-score Comparison Across Transformer Models", fontsize=18, weight="bold", pad=20)

# === 🔽 Step 4: Save as PNG ===
plt.tight_layout()
plt.savefig("results/macro_f1_comparison_table.png", dpi=300)
plt.show()
