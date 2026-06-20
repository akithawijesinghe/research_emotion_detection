import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load the correct cleaned dataset
df = pd.read_csv("data/cleaned_comments.csv")

# Extra safety cleaning
df = df.dropna(subset=['comment','emotion'])
df['comment'] = df['comment'].astype(str).str.strip()
df['emotion'] = df['emotion'].astype(str).str.lower().str.strip()

print("Final dataset shape:", df.shape)
print(df['emotion'].value_counts())

# Train-test split (stratified)
X = df['comment'].values
y = df['emotion'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# TF-IDF settings
tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    stop_words='english',
    lowercase=True
)

# Fit & transform
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save objects
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump((X_train, X_test, y_train, y_test), "models/raw_splits.joblib")
joblib.dump(X_train_tfidf, "models/X_train_tfidf.joblib")
joblib.dump(X_test_tfidf, "models/X_test_tfidf.joblib")

print("TF-IDF completed successfully!")
print("X_train_tfidf shape:", X_train_tfidf.shape)
print("X_test_tfidf shape:", X_test_tfidf.shape)
