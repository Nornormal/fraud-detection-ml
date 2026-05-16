# 💳 Credit Card Fraud Detection Using Machine Learning

A supervised machine learning project that builds and compares multiple classification models to detect fraudulent credit card transactions from a highly imbalanced financial dataset.

---

## Challenge

Financial institutions face a critical challenge: fraudulent transactions represent a tiny fraction of all activity — yet missing even one can result in significant financial loss. With a dataset severely skewed toward legitimate transactions (fraud cases under 1%), standard models tend to ignore the minority class entirely, producing high accuracy but dangerously low fraud recall. The core challenge was building a model that reliably catches fraud without flooding the system with false alarms.

---

## Solution

A full machine learning pipeline was developed in Python to preprocess, explore, and model the transaction data across six classifiers:

- **Data Cleaning:** Confirmed zero missing values and duplicates; removed the irrelevant `isFlaggedFraud` column.
- **Outlier Handling:** Applied winsorization (capping at the 5th–95th percentile) across all key financial features to reduce the influence of extreme transaction values.
- **EDA & Insights:** Discovered that fraud occurs exclusively in `CASH_OUT` and `TRANSFER` transaction types — a critical finding that directly informed feature selection.
- **Feature Engineering:** One-hot encoded transaction types, label-encoded categorical features, and applied MinMax scaling for distance-sensitive models.
- **Class Imbalance Handling:** Used `RandomUnderSampler` within a pipeline to balance training data for KNN, preventing the majority class from dominating learning.
- **Models Trained & Compared:**

| Model | Recall (Fraud) | Precision (Fraud) | F1-Score (Fraud) | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.92 | 0.03 | 0.06 | 0.96 |
| KNN (Baseline) | 0.67 | 0.91 | 0.77 | 0.99 |
| KNN + Undersampling | 0.92 | 0.06 | 0.12 | 0.98 |
| KNN + Scaled Data | 0.67 | 0.91 | 0.77 | 0.99 |
| Decision Tree (Balanced) | 0.78 | 0.81 | 0.79 | 0.99 |
| **Decision Tree + Scaled Data** | **0.85** | **0.85** | **0.85** | **0.99** |

- **Hyperparameter Tuning:** Applied `GridSearchCV` across regularization strengths, tree depth, and split criteria to find optimal configurations per model.

---

## Impact

The **Decision Tree with Scaled Data** emerged as the best-performing model, achieving an **F1-score of 0.85** and **99% accuracy** — balancing both fraud detection and false positive control effectively. Key business outcomes include:

- **High recall (0.85)** ensures the majority of actual fraud cases are flagged, reducing financial losses from undetected transactions.
- **High precision (0.85)** minimizes false positives, preventing legitimate customers from being incorrectly blocked.
- **Domain insight:** Fraud exclusively occurs in `CASH_OUT` and `TRANSFER` types — enabling rule-based pre-filtering before model scoring to reduce computational load in production.
- The pipeline design (scaling → balancing → tuning) provides a reusable template applicable to any imbalanced classification problem.

---

## Project Structure

```
fraud-detection-ml/
│
├── AIML Dataset.csv                               # Transaction dataset
├── Fraud_Detection_Machine_learning_models.ipynb  # Main analysis notebook
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Preprocessing, modeling & evaluation |
| imbalanced-learn | RandomUnderSampler for class balancing |
| SciPy | Normality testing (KS test) |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fraud-detection-ml.git
cd fraud-detection-ml

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy

# Launch the notebook
jupyter notebook Fraud_Detection_Machine_learning_models.ipynb
```

> **Note:** Place `AIML Dataset.csv` in the project root before running.

> **Dataset:** https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset
