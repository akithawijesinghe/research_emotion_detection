import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch.nn import CrossEntropyLoss
import evaluate

# Fix for macOS multiprocessing
if __name__ == '__main__':
    # Set environment variable to avoid tokenizer warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ------------------------
    # Load and Prepare Dataset
    # ------------------------
    df = pd.read_csv("data/prepared_data.csv")

with open("data/label_mappings.json") as f:
    label_data = json.load(f)
label2id = label_data["label2id"]
id2label = {int(k): v for k, v in label_data["id2label"].items()}

df["label"] = df["label"].astype(int)

# Check for class imbalance
print("Class distribution:")
print(df["label"].value_counts().sort_index())
print()

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# ------------------------
# Calculate Class Weights (for imbalanced datasets)
# ------------------------
class_counts = train_df['label'].value_counts().sort_index()
total_samples = len(train_df)
num_classes = len(class_counts)

# Calculate weights: inverse of frequency, normalized
class_weights = []
for i in range(num_classes):
    weight = total_samples / (num_classes * class_counts.get(i, 1))
    class_weights.append(weight)

class_weights = torch.FloatTensor(class_weights)
print(f"Class weights: {class_weights}")
print()

# ------------------------
# Tokenization with Increased Max Length
# ------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["comment"], 
        truncation=True, 
        padding=True, 
        max_length=256  # INCREASED from 128
    )

train_dataset = Dataset.from_pandas(train_df[["comment", "label"]]).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df[["comment", "label"]]).map(tokenize, batched=True)

# ------------------------
# Custom Trainer with Class Weights
# ------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ------------------------
# Load Model
# ------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id={v: int(k) for k, v in id2label.items()}
)

# Ensure all parameters are trainable
for param in model.parameters():
    param.requires_grad = True

# ------------------------
# Metrics
# ------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
        "precision_macro": precision.compute(predictions=preds, references=p.label_ids, average="macro")["precision"],
        "recall_macro": recall.compute(predictions=preds, references=p.label_ids, average="macro")["recall"]
    }

# ------------------------
# Improved Training Arguments
# ------------------------
training_args = TrainingArguments(
    output_dir="results/bert_finetune_improved",
    
    # Batch sizes and accumulation
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Effective batch size = 32
    
    # Training duration
    num_train_epochs=8,  # INCREASED from 6
    
    # Learning rate and schedule
    learning_rate=3e-5,  # INCREASED from 2e-5
    weight_decay=0.01,
    warmup_ratio=0.1,  # 10% of training for warmup
    lr_scheduler_type="cosine",  # Cosine decay instead of linear
    
    # Optimization
    fp16=False,  # Disabled for macOS MPS compatibility
    optim="adamw_torch",
    
    # Evaluation and saving
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,  # Load best model at end
    metric_for_best_model="f1_macro",  # Optimize for F1
    greater_is_better=True,
    
    # Logging
    logging_dir="logs",
    logging_steps=50,
    report_to="none",
    
    # Other
    seed=42,
    dataloader_num_workers=0,  # Changed to 0 for macOS compatibility
    push_to_hub=False,
)

# ------------------------
# Initialize Trainer
# ------------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    class_weights=class_weights  # Pass class weights
)

# ------------------------
# Train and Evaluate
# ------------------------
print("Starting training with improved hyperparameters...")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: ~{len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
print()

trainer.train()

# Final evaluation
print("\n" + "="*50)
print("FINAL EVALUATION RESULTS")
print("="*50)
metrics = trainer.evaluate()
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
print("="*50)

# Save the best model
trainer.save_model("results/bert_best_model")
tokenizer.save_pretrained("results/bert_best_model")
print("\n✅ Best model saved to 'results/bert_best_model'")

# Optional: Get predictions on validation set for analysis
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Save predictions for further analysis
results_df = pd.DataFrame({
    'true_label': true_labels,
    'pred_label': pred_labels,
    'comment': val_df['comment'].values
})
results_df.to_csv("results/validation_predictions.csv", index=False)
print("✅ Validation predictions saved to 'results/validation_predictions.csv'")