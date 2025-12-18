import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load TF-IDF training / test data
X_train_tfidf = joblib.load("models/X_train_tfidf.joblib")
X_test_tfidf = joblib.load("models/X_test_tfidf.joblib")
X_tr_raw, X_te_raw, y_train, y_test = joblib.load("models/raw_splits.joblib")

# Define classifier - use saga (multinomial) to avoid liblinear multiclass deprecation
clf = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=3000, random_state=42)

# Small, safe grid first
param_grid = {
    'C': [0.1, 1, 5],                  # try a small set first
    'class_weight': [None, 'balanced'] # balance option
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use built-in scoring string to avoid binary-scorer issues
scoring = 'f1_macro'

gs = GridSearchCV(clf, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=2, return_train_score=True)

print("Starting GridSearchCV...")
gs.fit(X_train_tfidf, y_train)

print("Best params:", gs.best_params_)
print("Best CV f1_macro:", gs.best_score_)

# Save best estimator
best = gs.best_estimator_
joblib.dump(best, "models/logreg_tuned.joblib")

# Save detailed CV results
cv_res = pd.DataFrame(gs.cv_results_)
cv_res.to_csv("results/logreg_gridsearch_results.csv", index=False)

# Evaluate best on test set
preds = best.predict(X_test_tfidf)
acc = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv("results/logreg_tuned_classification_report.csv")
cm = confusion_matrix(y_test, preds, labels=best.classes_)
pd.DataFrame(cm, index=best.classes_, columns=best.classes_).to_csv("results/logreg_tuned_confusion_matrix.csv")

print("Tuned model test accuracy:", acc)
print("Saved tuned model and results.")



