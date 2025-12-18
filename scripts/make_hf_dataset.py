# scripts/make_hf_dataset.py
import pandas as pd
from datasets import Dataset, ClassLabel
import os
import json

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/cleaned_comments.csv")

# Ensure columns
df = df.dropna(subset=['comment','emotion'])
df['comment'] = df['comment'].astype(str).str.strip()
df['emotion'] = df['emotion'].astype(str).str.lower().str.strip()

# map labels to ids
labels = sorted(df['emotion'].unique().tolist())
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

df['label'] = df['emotion'].map(label2id)

# Create HF dataset and stratified split using sklearn
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])

train_ds = Dataset.from_pandas(train_df[['comment','label']].reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df[['comment','label']].reset_index(drop=True))

# Save label maps and dataset to disk (optional)
with open("models/label2id.json", "w") as f:
    json.dump(label2id, f, indent=2)
with open("models/id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

train_ds.save_to_disk("data/hf_train_dataset")
test_ds.save_to_disk("data/hf_test_dataset")

print("Saved HF train/test datasets to data/hf_train_dataset and data/hf_test_dataset")
print("Labels:", labels)
