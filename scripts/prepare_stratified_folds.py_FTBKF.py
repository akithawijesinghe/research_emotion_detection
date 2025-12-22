# scripts/prepare_stratified_folds.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("data/prepared_data.csv")  # already label-encoded
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = fold

df.to_csv("data/stratified_data.csv", index=False)
print("✅ Added 'fold' column using StratifiedKFold")
