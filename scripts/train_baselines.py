# scripts/train_baselines.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import pandas as pd
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load feature matrices and label splits
X_train = joblib.load("models/X_train_tfidf.joblib")
X_test = joblib.load("models/X_test_tfidf.joblib")
X_tr_raw, X_te_raw, y_train, y_test = joblib.load("models/raw_splits.joblib")

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, C=1.0),
    "MultinomialNB": MultinomialNB(),
    "LinearSVC": LinearSVC(max_iter=20000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

results = []

# get labels order for consistent confusion matrices
labels = np.unique(y_test)

for name, model in models.items():
    print("Training:", name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    print(f"{name} accuracy: {acc:.4f}  macro-F1: {macro_f1:.4f}")

    clf_report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=labels)

    # Save model and results
    joblib.dump(model, f"models/{name}.joblib")
    pd.DataFrame(clf_report).transpose().to_csv(f"results/{name}_classification_report.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(f"results/{name}_confusion_matrix.csv")
    results.append((name, acc, macro_f1))

# Summary (accuracy + macro F1)
summary_df = pd.DataFrame(results, columns=["model", "accuracy", "macro_f1"])
summary_df.to_csv("results/model_comparison.csv", index=False)
print("All models trained and saved.")
print(summary_df)

# Save best model by macro F1
best_row = summary_df.sort_values("macro_f1", ascending=False).iloc[0]
best_model_name = best_row['model']
print("Best model by macro-F1:", best_model_name)
best_model = joblib.load(f"models/{best_model_name}.joblib")
joblib.dump(best_model, "models/best_model.joblib")
print("Saved best_model.joblib")
