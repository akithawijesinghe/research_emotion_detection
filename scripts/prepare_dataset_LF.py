# scripts/prepare_dataset_LF.py
import pandas as pd
import json

df = pd.read_csv("data/cleaned_comments.csv")
df = df.rename(columns={"emotion": "label"})
label_names = df['label'].unique().tolist()
label2id = {label: i for i, label in enumerate(sorted(label_names))}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['label'].map(label2id)
df.to_csv("data/prepared_data.csv", index=False)

with open("data/label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

print("✅ Dataset prepared and label mapping saved")



