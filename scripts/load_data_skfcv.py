# 1_load_data_skfcv.py
import pandas as pd
from datasets import Dataset

# Load your dataset
df = pd.read_csv("data/cleaned_comments.csv")

# Rename 'emotion' to 'label' and encode
df = df.rename(columns={"emotion": "label"})
label_names = df['label'].unique().tolist()
label2id = {label: i for i, label in enumerate(sorted(label_names))}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['label'].map(label2id)

# Save mappings for later use
import json
with open("data/label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

print("✅ Label encoding complete")
