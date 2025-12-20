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
