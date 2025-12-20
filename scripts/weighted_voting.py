import pandas as pd
import numpy as np
import os

# === 1️⃣ Load F1-based model weights ===
f1_table_path = "results/macro_f1_comparison_table.csv"
f1_df = pd.read_csv(f1_table_path)

f1_df["NormalizedWeight"] = f1_df["Avg"] / f1_df["Avg"].sum()
model_weights = {
    row["Model"].strip().lower(): row["NormalizedWeight"]
    for _, row in f1_df.iterrows()
}

print("📊 Model Weights (based on Avg F1):")
for name, weight in model_weights.items():
    print(f"  {name}: {weight:.4f}")

# === 2️⃣ Load softmax prediction files ===
pred_dir = "predictions"
model_file_map = {
    "bert": "bert_test_predictions.csv",
    "distilbert": "distilbert_test_predictions.csv",
    "roberta": "roberta_test_predictions.csv",
    "albert": "albert_test_predictions.csv",
}

model_prob_dfs = {}
expected_columns = None

for model, filename in model_file_map.items():
    path = os.path.join(pred_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")

    df = pd.read_csv(path)

    # Only keep numeric columns (probabilities)
    prob_df = df.select_dtypes(include=["number"])

    if expected_columns is None:
        expected_columns = list(prob_df.columns)
    elif list(prob_df.columns) != expected_columns:
        raise ValueError(
            f"❌ Column mismatch in {filename}:\n"
            f"  Expected: {expected_columns}\n"
            f"  Got     : {list(prob_df.columns)}"
        )

    model_prob_dfs[model] = prob_df.values
    print(f"✅ {filename} loaded: {prob_df.shape}")

# === 3️⃣ Weighted Voting ===
final_probs = np.zeros_like(next(iter(model_prob_dfs.values())))

for model, probs in model_prob_dfs.items():
    weight = model_weights[model]
    final_probs += weight * probs

# === 4️⃣ Final Prediction ===
final_preds = np.argmax(final_probs, axis=1)

# === 5️⃣ Save Output ===
output = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "prediction": final_preds
})
os.makedirs("results", exist_ok=True)
output.to_csv("results/ensemble_predictions.csv", index=False)

print("\n✅ Saved ensemble predictions to: results/ensemble_predictions.csv")
