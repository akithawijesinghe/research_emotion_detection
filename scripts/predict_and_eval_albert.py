# scripts/predict_and_eval_albert.py

import json, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_from_disk

MODEL_DIR = "models/albert_finetuned"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Load test dataset
test_ds = load_from_disk("data/hf_test_dataset")
comments = test_ds["comment"]
true_labels = test_ds["label"]

device = 0 if torch.cuda.is_available() else -1

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,
    device=device
)

id2label = json.load(open("models/id2label.json"))
label2id = {v: k for k, v in id2label.items()}

pred_names = []
pred_ids = []

# Predict
for c in comments:
    scores = clf(c, truncation=True, max_length=128)[0]
    best = max(scores, key=lambda x: x["score"])
    pred_names.append(best["label"])
    pred_ids.append(label2id[best["label"]])

# Save
df = pd.DataFrame({
    "comment": comments,
    "true_label": true_labels,
    "pred_label": pred_ids,
    "pred_label_name": pred_names,
})

df.to_csv("results/albert_test_predictions.csv", index=False)
print("Saved: results/albert_test_predictions.csv")
