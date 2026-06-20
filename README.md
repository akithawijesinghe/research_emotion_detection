# Emotion Detection in Crisis Social Media
### NLP Analysis of the 2019 Easter Sunday Attacks in Sri Lanka

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NLP](https://img.shields.io/badge/NLP-Emotion%20Detection-green)
![Transformers](https://img.shields.io/badge/Transformers-BERT-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Research](https://img.shields.io/badge/Research-NLP-purple)
![University](https://img.shields.io/badge/University-University%20of%20Vavuniya-orange)

---

## Overview

This research project presents a Natural Language Processing (NLP) framework for detecting emotions in crisis-related social media content. The study focuses on YouTube comments associated with the **2019 Easter Sunday Attacks in Sri Lanka** and aims to automatically classify emotional expressions into multiple psychologically relevant categories.

The project combines traditional machine learning techniques and state-of-the-art transformer-based deep learning models to understand public emotional responses during national crises.

---

## Research Motivation

During crisis events, social media platforms become major channels where people express emotions, share experiences, seek information, and react to unfolding situations.

Understanding these emotional responses can support:

- Crisis communication
- Public sentiment monitoring
- Mental health awareness
- Emergency response planning
- Social research and policy making

Manual analysis of thousands of comments is impractical, making NLP-based automated emotion detection a valuable solution.

---

## Research Objectives

### Main Objective

Develop and evaluate an NLP-based framework capable of identifying emotions expressed in crisis-related YouTube comments.

### Specific Objectives

- Collect crisis-related YouTube comments
- Manually annotate emotional labels
- Preprocess social media text
- Extract textual features
- Compare traditional ML and Transformer models
- Evaluate model performance
- Deploy the best-performing model for real-time emotion prediction

---

# Dataset

## Data Source

YouTube comments related to the **2019 Easter Sunday Attacks in Sri Lanka**

### Collection Method

- YouTube Data API v3
- English-language comments
- Crisis-related videos

### Dataset Size

Approximately:

```text
12,500 Comments
```

### Emotion Categories

The dataset was manually annotated into five emotion classes:

| Label | Emotion |
|---------|---------|
| A | Anger |
| AN | Anxiety |
| F | Fear |
| G | Grief |
| S | Sadness |

---

## Annotation Process

The dataset was manually labeled by three human annotators.

### Inter-Annotator Agreement

```text
Fleiss' Kappa = 0.52
```

This indicates moderate agreement among annotators.

---

# Methodology

## Research Pipeline

```text
YouTube Data Collection
          │
          ▼
Manual Annotation
          │
          ▼
Text Preprocessing
          │
          ▼
Feature Extraction
          │
          ▼
Model Training
          │
          ▼
Model Evaluation
          │
          ▼
Real-Time Deployment
```

---

# Data Preprocessing

The following preprocessing techniques were applied:

- Lowercasing
- URL Removal
- Emoji Removal
- Punctuation Removal
- Tokenization
- Stopword Removal
- Lemmatization
- Contraction Expansion
- Whitespace Normalization

---

# Feature Extraction

## Traditional Features

### TF-IDF

- Term Frequency
- Inverse Document Frequency
- Sparse vector representation

---

## Transformer Embeddings

Contextual embeddings generated using:

- BERT
- RoBERTa
- DistilBERT
- ALBERT

---

# Models Evaluated

## Traditional Machine Learning Models

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

---

## Transformer Models

- BERT
- RoBERTa
- DistilBERT
- ALBERT

---

# Experimental Results

## Best Performing Model

```text
Fine-Tuned BERT
```

### Performance

| Metric | Score |
|----------|----------|
| Macro F1 Score | 0.8133 |

The fine-tuned BERT model achieved the best overall performance by effectively capturing contextual emotional expressions in crisis-related text.

---

# Repository Structure

```text
emotion_detection/
│
├── data/
│   ├── raw datasets
│   ├── processed datasets
│   └── Hugging Face datasets
│
├── scripts/
│   ├── preprocessing
│   ├── feature extraction
│   ├── training
│   └── evaluation
│
├── Main Scripts/
│   ├── Ycomments.py
│   ├── load_and_check.py
│   ├── feature_extraction.py
│   ├── train_baselines.py
│   ├── train_bert.py
│   ├── prepare_dataset_LF.py
│   ├── make_hf_dataset.py
│   └── tokenize_fold_data.py
│
├── results/
│   ├── model outputs
│   ├── evaluation reports
│   └── visualizations
│
├── predictions/
│
├── share/
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/research_emotion_detection.git

cd research_emotion_detection
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Data Collection

```bash
python "Main Scripts/Ycomments.py"
```

---

## Data Verification

```bash
python "Main Scripts/load_and_check.py"
```

---

## Feature Extraction

```bash
python "Main Scripts/feature_extraction.py"
```

---

## Train Baseline Models

```bash
python "Main Scripts/train_baselines.py"
```

---

## Train BERT Model

```bash
python "Main Scripts/train_bert.py"
```

---

# Technologies Used

## Programming Language

- Python

## NLP Libraries

- NLTK
- Hugging Face Transformers
- Datasets

## Machine Learning

- Scikit-learn

## Deep Learning

- PyTorch

## Data Processing

- Pandas
- NumPy

## Deployment

- Streamlit
- Docker
- Hugging Face Spaces

## Data Collection

- YouTube Data API v3

---

# Research Contributions

### Dataset Contribution

Created one of the first manually annotated Sri Lankan crisis-related emotion datasets.

### Comparative Analysis

Compared traditional machine learning models with transformer architectures.

### Fine-Tuning Study

Evaluated the effectiveness of transformer fine-tuning for crisis emotion classification.

### Real-Time Application

Developed an interactive emotion detection system.

### Crisis Communication Insights

Provided a data-driven understanding of emotional responses during the Easter Sunday attacks.

---

# Future Improvements

- Support Sinhala and Tamil languages
- Multi-label emotion classification
- Larger crisis datasets
- Social media platform expansion
- Explainable AI integration
- Real-time social media monitoring dashboards

---

# Research Team

### Group 01

| Registration No | Name |
|-----------------|---------|
| 2019ICTS74 | A.M.F. Farija |
| 2019ICTS103 | A.W. Wijesinghe |
| 2019ICTS122 | R.F. Faisana |

### Supervisor

Ms. W.A.S.C. Perera

Department of Information and Communication Technology

Faculty of Technological Studies

University of Vavuniya

---

# Citation

If you use this work in your research, please cite:

```bibtex
@thesis{emotion_detection_2025,
  title={Emotion Detection in Crisis Social Media: An NLP Analysis of the 2019 Easter Sunday Attacks in Sri Lanka},
  author={Farija, A.M.F. and Wijesinghe, A.W. and Faisana, R.F.},
  year={2025},
  school={University of Vavuniya}
}
```

---

# Acknowledgements

We would like to thank:

- University of Vavuniya
- Faculty of Technological Studies
- Department of Information and Communication Technology
- Ms. W.A.S.C. Perera (Supervisor)
- All annotators and contributors involved in this research

---

## License

This project is developed for academic and research purposes.

© 2025 Group 01 — University of Vavuniya