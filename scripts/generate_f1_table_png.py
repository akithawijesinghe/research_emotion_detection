import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
RESULTS_DIR = "results"
MODEL_REPORTS = {
    "BERT": "bert_classification_report.csv",
    "DistilBERT": "distilbert_classification_report.csv",
    "RoBERTa": "roberta_classification_report.csv",
    "ALBERT": "albert_classification_report.csv",
}

EMOTIONS = ["anger", "anxiety", "fear", "grief", "sadness"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# LOAD LABEL MAP
# =========================
id2label = json.load(open("models/id2label.json"))
id2label = {int(k): v.lower() for k, v in id2label.items()}

# =========================
# BUILD F1 TABLE
# =========================
rows = []

for model, file in MODEL_REPORTS.items():
    print(f"Processing {model}...")

    df = pd.read_csv(os.path.join(RESULTS_DIR, file))
    df.columns = [c.strip().lower() for c in df.columns]
    label_col = df.columns[0]

    df[label_col] = df[label_col].astype(str).str.strip()
    df = df.set_index(label_col)

    row = {"Model": model}
    f1_scores = []

    for label_id, emotion in id2label.items():
        if emotion not in EMOTIONS:
            continue

        # Accept both numeric-label and text-label reports
        key = str(label_id) if str(label_id) in df.index else emotion
        f1 = float(df.loc[key, "f1-score"])

        row[emotion.capitalize()] = round(f1, 3)
        f1_scores.append(f1)

    row["Avg"] = round(np.mean(f1_scores), 3)
    rows.append(row)

final_df = pd.DataFrame(rows)
final_df.to_csv(os.path.join(RESULTS_DIR, "macro_f1_comparison_table.csv"), index=False)

print("Saved macro_f1_comparison_table.csv")

# =========================
# CREATE PAPER-STYLE TABLE IMAGE
# =========================
fig, ax = plt.subplots(figsize=(11, 3))
ax.axis("off")

table = ax.table(
    cellText=final_df.values,
    colLabels=final_df.columns,
    cellLoc="center",
    loc="center",
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

# Remove all borders
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_linewidth(0)

# Bold header
for col in range(len(final_df.columns)):
    table[(0, col)].set_text_props(weight="bold")

# Booktabs-style rules
n_cols = len(final_df.columns)
last_row = len(final_df)

# Top rule
for c in range(n_cols):
    cell = table[(0, c)]
    cell.visible_edges = "T"
    cell.set_edgecolor("black")
    cell.set_linewidth(1.2)

# Mid rule
for c in range(n_cols):
    cell = table[(1, c)]
    cell.visible_edges = "T"
    cell.set_edgecolor("black")
    cell.set_linewidth(0.8)

# Bottom rule
for c in range(n_cols):
    cell = table[(last_row, c)]
    cell.visible_edges = "B"
    cell.set_edgecolor("black")
    cell.set_linewidth(1.2)

plt.title(
    "Macro F1-score Comparison Across Transformer Models",
    fontsize=13,
    pad=10,
)

plt.savefig(
    os.path.join(RESULTS_DIR, "macro_f1_comparison.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("✅ Saved macro_f1_comparison.png")
