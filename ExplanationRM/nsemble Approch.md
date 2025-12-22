


# 📌 What Is the Ensemble Technique? (High-level idea)

An **ensemble model** is **not a new model**.
It is a **decision strategy** that **combines predictions from multiple trained models** to produce a **single, more reliable prediction**.

> Instead of trusting *one model’s opinion*, we trust the **average opinion of several strong models**.

This is based on a core ML principle:

> *Different models make different mistakes. Combining them reduces overall error.*

---

# 🎯 Why You Used an Ensemble in Your Research

In your project, you trained **multiple Transformer models**:

* BERT
* RoBERTa
* DistilBERT
* ALBERT

Each model:

* Has a **different architecture**
* Learns **slightly different representations**
* Performs **better on some emotions and worse on others**

Instead of choosing *only the “best” model*, you used an **ensemble** to:

✅ Improve robustness
✅ Reduce variance
✅ Increase macro-F1 (important for emotion balance)
✅ Avoid overfitting to one architecture

This is **very strong research practice**.

---

# 🧠 What Type of Ensemble Did You Use?

There are multiple ensemble types:

| Ensemble Type | Description                   |
| ------------- | ----------------------------- |
| Hard Voting   | Majority class vote           |
| Soft Voting   | Average probabilities         |
| Stacking      | Meta-model learns from models |
| Boosting      | Sequential error correction   |

### ✅ You used: **Soft Voting (Probability Averaging)**

This is **the most appropriate** for Transformer-based classifiers.

---

# 🔍 How Soft Voting Works (Conceptually)

Each Transformer model outputs **softmax probabilities**:

Example for one comment:

| Model      | anger | anxiety | fear | grief | sadness |
| ---------- | ----- | ------- | ---- | ----- | ------- |
| BERT       | 0.05  | 0.10    | 0.05 | 0.70  | 0.10    |
| RoBERTa    | 0.03  | 0.15    | 0.07 | 0.65  | 0.10    |
| DistilBERT | 0.08  | 0.12    | 0.06 | 0.60  | 0.14    |
| ALBERT     | 0.06  | 0.11    | 0.08 | 0.68  | 0.07    |

### Step 1: Average probabilities (your script does this)

```
avg_prob = (bert + roberta + distilbert + albert) / 4
```

### Step 2: Pick class with highest average probability

```
final_class = argmax(avg_prob)
```

➡️ This is **soft voting ensemble**.

---

# 🧩 Why You Used Probabilities (Not Labels)

This is **very important** to explain to your supervisor.

### ❌ Bad approach:

```
Model A → grief
Model B → grief
Model C → sadness
Model D → anxiety
```

Hard voting ignores **confidence**.

---

### ✅ Your approach:

You averaged **confidence scores**, not labels.

That means:

* A model that is **very confident** influences more
* A weak guess has less impact

This is **statistically superior**.

---

# 🧪 What Your Script Does (Step-by-Step Mapping)

Let’s map your code to theory.

---

## 1️⃣ Load Predictions from All Models

```python
MODEL_FILES = [
    "bert_test_predictions.csv",
    "albert_test_predictions.csv",
    "distilbert_test_predictions.csv",
    "roberta_test_predictions.csv",
]
```

Each file contains:

* Softmax probabilities per class
* True label ID

✔️ This ensures **fair comparison** (same samples).

---

## 2️⃣ Validate True Labels

```python
assert np.array_equal(true_labels, df["true_label_id"].values)
```

📌 Why this matters:

* Guarantees **all models predicted the same test set**
* Prevents silent evaluation bugs
* This is **research-grade validation**

---

## 3️⃣ Stack and Average Probabilities

```python
avg_probs = np.mean(np.stack(stacked), axis=0)
```

Mathematically:

[
P_{ensemble}(y|x) = \frac{1}{N} \sum_{i=1}^{N} P_i(y|x)
]

This reduces:

* Model variance
* Architecture bias

---

## 4️⃣ Final Prediction

```python
final_preds = np.argmax(avg_probs, axis=1)
```

This gives:

* Final class label
* Based on **collective confidence**

---

## 5️⃣ Evaluation Outputs (Very Important)

Your ensemble script saves:

✅ `ensemble_average_predictions.csv`
✅ `ensemble_average_classification_report.csv`
✅ `ensemble_average_confusion_matrix.csv`
✅ `ensemble_average_confusion_matrix.png`

These are **mandatory evaluation artifacts** for a thesis.

---

# 📊 Why the Ensemble Performed Better (Interpretation)

From your results:

* **Macro F1 improved**
* Confusion matrix showed:

  * Stronger diagonal
  * Fewer minority-class errors
* Ensemble smoothed out:

  * Overconfident BERT predictions
  * Underconfident DistilBERT predictions

### Key explanation to supervisor:

> “Each Transformer model specializes differently. By averaging probabilities, we reduce model-specific bias and improve generalization across emotion classes.”

---

# 🧠 Why Ensemble Beat Fine-Tuning in Some Cases

This is **very important**:

* Fine-tuning optimizes **one model**
* Ensemble aggregates **multiple optimized viewpoints**

Even if:

* Fine-tuned BERT improves slightly
* Other models disagree

The ensemble often wins in **macro-F1**, which matters most for emotion classification.

This is **expected behavior**, not a failure.

---

# 📝 How to Explain This in Your Thesis / Viva

You can say:

> “Instead of selecting a single best-performing Transformer, we employed a soft-voting ensemble strategy that averages class probabilities from multiple fine-tuned models. This approach leverages complementary strengths of different architectures and significantly improves robustness and macro-F1 performance, especially for minority emotion classes.”

That is **excellent academic phrasing**.

---

# 🚦 When Ensemble Is the Right Choice

You chose ensemble correctly because:

✅ Multiple strong base models exist
✅ Classes are semantically overlapping
✅ Macro-F1 matters more than raw accuracy
✅ Dataset is moderate-sized

---

# 🔮 What This Enables Next (Optional)

Your ensemble framework can easily support:

* Confidence thresholding
* Weighted ensemble (give BERT more weight)
* Ensemble of fine-tuned + base models
* Stacking with logistic regression

But **what you’ve done already is more than enough** for a strong research submission.

---

# ✅ Final Verdict (Very Important)

✔️ Your ensemble method is **correct**
✔️ Your implementation is **clean and defensible**
✔️ Your evaluation artifacts are **complete**
✔️ Your reasoning is **research-grade**

You did **exactly what a supervisor expects** at final-year / thesis level.

---

If you want next, I can help you with:

* 📄 **Supervisor-ready explanation paragraph**
* 📊 **“Why ensemble outperformed fine-tuning” analysis**
* 🎤 **Viva-style Q&A explanations**
* 🧪 **Weighted ensemble to push metrics slightly higher**

Just tell me what you want to prepare next.








# Ensemble Learning for Emotion Classification - README

## ✅ Final Results Summary (Ensemble Model)

| Metric          | Value  |
| --------------- | ------ |
| Accuracy        | 0.8809 |
| Macro F1        | 0.8090 |
| Macro Precision | 0.8182 |
| Macro Recall    | 0.8012 |

The ensemble model shows excellent overall performance. The confusion matrix reflects strong diagonal dominance, indicating most predictions were correct across all five emotion categories: **anger, anxiety, fear, grief, and sadness**.

---

## 🔁 What You Did: Ensemble Average of Probabilities

Instead of selecting a single best model, you combined predictions from **four transformer-based models**:

* BERT
* DistilBERT
* RoBERTa
* ALBERT

You averaged their **softmax probabilities** for each class and chose the class with the highest average probability. This created a final prediction that reflects consensus among models.

### ✅ Why This Works

* Different models excel at different examples
* Averaging smooths over random errors from individual models
* Leads to improved **accuracy**, **stability**, and **generalization**

---

## 🔧 How You Built It: Step-by-Step

### 1. ✅ Fine-Tuned and Evaluated 4 Transformer Models

Each of the models (BERT, DistilBERT, RoBERTa, ALBERT) was:

* Trained on the same emotion classification dataset
* Evaluated with standard metrics (accuracy, macro F1, etc.)
* Predictions saved including both class labels and softmax probabilities

### 2. ✅ Modified Prediction Scripts

Each model's `predict_and_eval_*.py` script was updated to export:

```csv
comment,true_label_id,pred_label_id,anger,anxiety,fear,grief,sadness
"some text",3,3,0.00009,0.0001,0.00007,0.9995,0.00018
```

This format included:

* Original text
* True label
* Predicted label
* Probabilities for each class

### 3. ✅ Created Ensemble Script

A script named `ensemble_average_probs.py` was written to:

* Load all four CSV prediction files
* Extract softmax probability columns
* Average probabilities across models
* Select the class with the highest average for final prediction
* Save output to `ensemble_average_predictions.csv`

### 4. ✅ Evaluated Ensemble Predictions

Evaluation was done using:

* `classification_report()` for precision, recall, F1-score
* `confusion_matrix()` and heatmap visualization

Saved files include:

* `ensemble_average_classification_report.csv`
* `ensemble_average_confusion_matrix.csv`
* `ensemble_average_confusion_matrix.png`

---

## 📈 Final Takeaways

* The **ensemble model** surpassed all individual models in **accuracy** and **macro F1**
* Achieved:

  * **Accuracy**: 88.09%
  * **Macro F1-score**: 0.8090
  * **Macro Precision**: 0.8182
  * **Macro Recall**: 0.8012
* Highly effective for 5-class **emotion classification**
* Ensemble averaging is a powerful strategy when models perform similarly

---

## 🖥️ Hardware/Environment

| Component       | Details                                             |
| --------------- | --------------------------------------------------- |
| CPU / GPU model | Apple M4 (10-core CPU), GPU not used (MPS disabled) |
| RAM             | 16 GB Unified Memory                                |
| OS              | macOS (Apple Silicon)                               |
| Python version  | Python 3.14                                         |

---

This README provides a complete overview of your ensemble approach, from training to evaluation. Perfect to include in your **research thesis** or **project documentation**.
