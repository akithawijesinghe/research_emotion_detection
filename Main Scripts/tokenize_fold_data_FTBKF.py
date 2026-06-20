# scripts/tokenize_fold_data.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

df = pd.read_csv("data/stratified_data.csv")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["comment"], truncation=True, padding="max_length", max_length=128)

# Change this value for other folds
fold = 0  
train_df = df[df["fold"] != fold]
val_df = df[df["fold"] == fold]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["comment", "fold", "__index_level_0__"])
val_dataset = val_dataset.remove_columns(["comment", "fold", "__index_level_0__"])
train_dataset.set_format("torch")
val_dataset.set_format("torch")
