# # scripts/ensemble_average_probs.py

# import os
# import pandas as pd
# import numpy as np

# # === 1️⃣ Define prediction sources ===
# PREDICTIONS_DIR = "predictions"
# OUTPUT_FILE = "results/ensemble_average_predictions.csv"

# model_files = {
#     "bert": "bert_test_predictions.csv",
#     "distilbert": "distilbert_test_predictions.csv",
#     "albert": "albert_test_predictions.csv",
#     "roberta": "roberta_test_predictions.csv",
# }

# # === 2️⃣ Load softmax probabilities ===
# all_probs = []
# label_cols = None
# true_labels = None

# for name, filename in model_files.items():
#     path = os.path.join(PREDICTIONS_DIR, filename)
#     df = pd.read_csv(path)

#     # Extract probability columns
#     if label_cols is None:
#         label_cols = [col for col in df.columns if col not in {"comment", "true_label", "pred_label", "pred_label_name", "true_label_id", "pred_label_id"}]

#     probs = df[label_cols].values
#     all_probs.append(probs)

#     # Sanity check: all true labels should match
#     if true_labels is None:
#         true_labels = df["true_label_id"].values
#     else:
#         assert np.array_equal(true_labels, df["true_label_id"].values), f"❌ True labels mismatch in {filename}"

# print(f"✅ Loaded softmax probabilities from {len(all_probs)} models.")

# # === 3️⃣ Average probabilities and get predictions ===
# mean_probs = np.mean(np.array(all_probs), axis=0)
# final_preds = np.argmax(mean_probs, axis=1)

# # === 4️⃣ Save final predictions ===
# output_df = pd.DataFrame({
#     "id": np.arange(len(final_preds)),
#     "true_label": true_labels,
#     "pred_label": final_preds,
# })

# output_df.to_csv(OUTPUT_FILE, index=False)
# print(f"✅ Ensemble prediction saved to: {OUTPUT_FILE}")




# scripts/ensemble_average_probs.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 1️⃣ Paths ===
PRED_DIR = "predictions"
OUT_PATH = "results/ensemble_average_predictions.csv"
MODEL_FILES = [
    "bert_test_predictions.csv",
    "albert_test_predictions.csv",
    "distilbert_test_predictions.csv",
    "roberta_test_predictions.csv",
]

# === 2️⃣ Load and Validate Data ===
dfs = []
for fname in MODEL_FILES:
    path = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {fname}")
    
    df = pd.read_csv(path)

    # Ensure true label column exists
    true_label_col = None
    for col in ["true_label_id", "true_label", "label"]:
        if col in df.columns:
            true_label_col = col
            break
    if true_label_col is None:
        raise ValueError(f"❌ True label column not found in {fname}")

    df = df[["anger", "anxiety", "fear", "grief", "sadness", true_label_col]]
    dfs.append(df)

print("✅ All model prediction files loaded.")

# === 3️⃣ Average Probabilities ===
prob_cols = ["anger", "anxiety", "fear", "grief", "sadness"]
stacked = [df[prob_cols].values for df in dfs]
avg_probs = np.mean(np.stack(stacked), axis=0)

# === 4️⃣ Final Predictions ===
final_preds = np.argmax(avg_probs, axis=1)
label_cols = prob_cols  # class names
id2label = {i: label for i, label in enumerate(label_cols)}
pred_names = [id2label[i] for i in final_preds]

# === 5️⃣ Save Final Ensemble Predictions ===
# Use true labels from the first file
true_labels = dfs[0][true_label_col].tolist()

out_df = pd.DataFrame(avg_probs, columns=label_cols)
out_df["true_label_id"] = true_labels
out_df["pred_label_id"] = final_preds
out_df.to_csv(OUT_PATH, index=False)

print(f"✅ Saved ensemble predictions to: {OUT_PATH}")

# === 6️⃣ Evaluate Performance ===
report = classification_report(true_labels, final_preds, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("results/ensemble_average_classification_report.csv")
print("✅ Classification report saved.")

# Confusion matrix
cm = confusion_matrix(true_labels, final_preds)
pd.DataFrame(cm, index=label_cols, columns=label_cols).to_csv("results/ensemble_average_confusion_matrix.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_cols, yticklabels=label_cols)
plt.title("Ensemble Average Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/ensemble_average_confusion_matrix.png")
plt.close()
print("✅ Confusion matrix plot saved.")
