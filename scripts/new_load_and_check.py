import pandas as pd

# change path if necessary
df = pd.read_csv("data/new_cleaned_comments.csv")

#Drop Null comments & emotions
df = df.dropna(subset=['comment'])
df = df.dropna(subset=['emotion'])

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df['emotion'].value_counts())
print("\nSample rows:")
print(df.sample(10, random_state=42).to_string(index=False))
# Quick check for nulls
print("\nNull counts:\n", df.isnull().sum())

# Save a small sample for manual review if needed
df.sample(200, random_state=42).to_csv("results/sample_200_for_review.csv", index=False)




