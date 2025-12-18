# scripts/extract_misclassified.py
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer

os.makedirs("results", exist_ok=True)

# Load objects
best = joblib.load("models/best_model.joblib")            # best saved model
X_train_raw, X_test_raw, y_train, y_test = joblib.load("models/raw_splits.joblib")
X_test_tfidf = joblib.load("models/X_test_tfidf.joblib")

# Predict
preds = best.predict(X_test_tfidf)

# If model supports predict_proba, get probabilities; else None
probs = None
if hasattr(best, "predict_proba"):
    probs = best.predict_proba(X_test_tfidf)

# Find misclassified indices
mis_idx = np.where(preds != y_test)[0]
print("Total misclassified:", len(mis_idx))

# sample up to 100
rng = np.random.RandomState(42)
take = min(100, len(mis_idx))
sample_idx = rng.choice(mis_idx, size=take, replace=False)

rows = []
for i in sample_idx:
    row = {
        "text": X_test_raw[i],
        "true": y_test[i],
        "pred": preds[i],
        "text_length": len(X_test_raw[i].split())
    }
    # include top probability if available
    if probs is not None:
        # convert probs to dict label->prob
        labels = best.classes_
        prob_dict = {labels[j]: float(probs[i, j]) for j in range(len(labels))}
        row.update({"probs": prob_dict})
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results/misclassified_sample_100.csv", index=False)
print("Saved results/misclassified_sample_100.csv")
