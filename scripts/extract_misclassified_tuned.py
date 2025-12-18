# scripts/extract_misclassified_tuned.py
import joblib
import pandas as pd
import numpy as np
import os

os.makedirs("results", exist_ok=True)

# Load tuned model + raw splits + tfidf test matrix
tuned = joblib.load("models/logreg_tuned.joblib")          # tuned model
X_train_raw, X_test_raw, y_train, y_test = joblib.load("models/raw_splits.joblib")
X_test_tfidf = joblib.load("models/X_test_tfidf.joblib")

# Predict
preds = tuned.predict(X_test_tfidf)

# If predict_proba available, get it (for LogisticRegression it is)
probs = None
if hasattr(tuned, "predict_proba"):
    probs = tuned.predict_proba(X_test_tfidf)

# misclassified indices
mis_idx = np.where(preds != y_test)[0]
print("Total misclassified:", len(mis_idx))

# sample up to 100
rng = np.random.RandomState(42)
take = min(100, len(mis_idx))
sample_idx = rng.choice(mis_idx, size=take, replace=False)

rows = []
labels = tuned.classes_
for i in sample_idx:
    row = {
        "text": X_test_raw[i],
        "true": y_test[i],
        "pred": preds[i],
        "text_length": len(X_test_raw[i].split())
    }
    if probs is not None:
        probs_i = {labels[j]: float(probs[i, j]) for j in range(len(labels))}
        row["probs"] = probs_i
        # add top predicted probability
        row["pred_prob"] = float(probs[i, list(labels).index(preds[i])])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results/misclassified_sample_100_tuned.csv", index=False)
print("Saved results/misclassified_sample_100_tuned.csv")
