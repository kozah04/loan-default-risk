# Loan Default Risk Model

**Project 4 of 10 — Gwachat Kozah's Data Science Portfolio**

A full machine learning pipeline to predict loan default risk using data from SuperLender, a Nigerian digital lending company. The model predicts whether a repeat loan applicant will repay on time (Good) or default (Bad), based on their demographic profile and prior loan repayment behaviour.

---

## Problem Statement

SuperLender needs to assess credit risk at the point of each loan application. For repeat customers — those who have taken prior loans — behavioural repayment history is available and can be incorporated into the risk model alongside demographic features. The task is binary classification: predict Good (1) or Bad (0).

---

## Dataset

The data is sourced from the [Data Science Nigeria Loan Default Prediction Challenge](https://zindi.africa/competitions/data-science-nigeria-challenge-1-loan-default-prediction) on Zindi.

The dataset is the sole property of Zindi and the competition host and cannot be redistributed. To reproduce this work, download the data directly from the Zindi competition page and place the files in `data/raw/`.

The dataset consists of three linked tables:

| File | Description |
|------|-------------|
| `trainperf.csv` | Loan performance data including the target variable `good_bad_flag` |
| `traindemographics.csv` | Customer demographic data |
| `trainprevloans.csv` | All prior loans taken by each customer before the loan to be predicted |
| `testperf.csv` | Performance data for the test set (no target) |
| `testdemographics.csv` | Demographic data for the test set |
| `testprevloans.csv` | Prior loan history for test set customers |

---

## Project Structure

```
loan-default-risk/
├── data/
│   ├── raw/              # Original Zindi files — gitignored
│   └── processed/        # Merged and cleaned datasets — gitignored
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modelling_and_evaluation.ipynb
├── src/
│   ├── loader.py         # Data loading and merging
│   ├── features.py       # Feature engineering
│   ├── model.py          # Training, tuning, and inference
│   └── evaluate.py       # Metrics, threshold optimisation, plots
├── tests/
│   ├── test_loader.py
│   ├── test_features.py
│   └── test_model.py
├── outputs/
│   ├── models/           # Saved model files — gitignored
│   ├── figures/          # Generated plots — gitignored
│   └── submissions/      # Zindi submission CSVs
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

1. Data loading and merging across three tables
2. Exploratory data analysis
3. Feature engineering — behavioural features derived from prior loan history
4. Baseline model — Logistic Regression
5. Imbalanced class handling — class weights and SMOTE
6. Model comparison — Logistic Regression, Random Forest, XGBoost
7. Hyperparameter tuning — RandomizedSearchCV
8. Threshold optimisation — balancing precision and recall for business context
9. Model explainability — SHAP feature importance

---

## Setup

```bash
conda env create -f environment.yml
conda activate loan-default-risk
```

---

## Key Results

*To be updated after modelling is complete.*

---

## Skills Demonstrated

- Multi-table data merging and feature engineering
- Handling class imbalance (SMOTE, class weights, threshold tuning)
- Model comparison and hyperparameter tuning
- SHAP explainability
- Credit risk modelling concepts (willingness to pay, ability to pay)

---

## Author

**Gwachat Kozah**
[github.com/kozah04](https://github.com/kozah04) | gwachatkozah04@gmail.com
