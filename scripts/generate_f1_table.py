import pandas as pd
import numpy as np
import json

# ==============================
# Load label mapping
# ==============================
id2label = json.load(open("models/id2label.json"))
id2label = {int(k): v.lower() for k, v in id2label.items()}

EMOTIONS = ["anger", "anxiety", "fear", "grief", "sadness"]

MODELS = {
    "BERT": "results/bert_classification_report.csv",
    "DistilBERT": "results/distilbert_classification_report.csv",
    "RoBERTa": "results/roberta_classification_report.csv",
    "ALBERT": "results/albert_classification_report.csv",
}

rows = []

for model_name, path in MODELS.items():
    print(f"\nProcessing {model_name} → {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # First column is label
    label_col = df.columns[0]
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df.set_index(label_col)

    row = {"MODEL": model_name}
    f1_values = []

    for label_id, emotion in id2label.items():
        # Decide lookup key
        if str(label_id) in df.index:
            key = str(label_id)          # numeric-label report
        elif emotion in df.index:
            key = emotion                # text-label report
        else:
            raise ValueError(
                f"{model_name}: neither label '{label_id}' nor '{emotion}' found.\n"
                f"Available rows: {list(df.index)}"
            )

        f1 = float(df.loc[key, "f1-score"])
        row[emotion.upper()] = round(f1, 3)
        f1_values.append(f1)

    row["AVG"] = round(np.mean(f1_values), 3)
    rows.append(row)

# ==============================
# Save final comparison table
# ==============================
final_df = pd.DataFrame(rows)
final_df.to_csv("results/model_f1_comparison_table.csv", index=False)

print("\n✅ SUCCESS!")
print("Saved → results/model_f1_comparison_table.csv\n")
print(final_df)
