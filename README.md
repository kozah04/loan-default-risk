# Loan Default Risk Model

A full machine learning pipeline to predict loan default risk using data from SuperLender, a Nigerian digital lending company. The model predicts whether a repeat loan applicant will repay on time (Good) or default (Bad), based on their demographic profile and prior repayment behaviour.

**Best result: 0.2076 leaderboard error rate (48% improvement over baseline)**

---

## Problem Statement

SuperLender assesses credit risk at the point of each loan application. For repeat customers, prior repayment history is available and incorporated into the risk model alongside demographics. The task is binary classification: predict Good (1) or Bad (0).

The dataset is heavily imbalanced at 78.2% Good and 21.8% Bad. Missing a defaulter costs more than a false alarm, so the model is optimised for recall on the Bad class throughout.

---

## Dataset

Sourced from the [Data Science Nigeria Loan Default Prediction Challenge](https://zindi.africa/competitions/data-science-nigeria-challenge-1-loan-default-prediction) on Zindi.

The data is the sole property of Zindi and cannot be redistributed. To reproduce this work, download the files from the Zindi competition page and place them in `data/raw/`.

| File | Description |
|------|-------------|
| `trainperf.csv` | Loan performance data including the target `good_bad_flag` |
| `traindemographics.csv` | Customer demographics |
| `trainprevloans.csv` | All prior loans per customer before the loan to predict |
| `testperf.csv` | Test performance data (no target) |
| `testdemographics.csv` | Test demographics |
| `testprevloans.csv` | Prior loan history for test customers |

---

## Project Structure

```
loan-default-risk/
├── data/
│   ├── raw/              # Zindi files — gitignored
│   └── processed/        # Merged datasets — gitignored
├── notebooks/
│   ├── eda.ipynb         # Exploratory data analysis
│   └── modelling_and_evaluation.ipynb
├── src/
│   ├── loader.py         # Data loading and merging
│   ├── features.py       # Feature engineering
│   ├── model.py          # Training, tuning, stacking, calibration
│   └── evaluate.py       # Metrics, threshold search, SHAP plots
├── tests/
│   ├── test_loader.py
│   ├── test_features.py
│   └── test_model.py
├── outputs/
│   ├── models/           # Saved models — gitignored
│   ├── figures/          # Generated plots — gitignored
│   └── submission/       # Best Zindi submission
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Pipeline

1. Load and merge three tables into one row per customer
2. Time-based train/validation split (oldest 80% train, newest 20% validate)
3. Feature engineering — behavioural aggregations, recency features, PLTR interaction features
4. SMOTE ablation — find the best imbalance strategy per model
5. Model comparison and hyperparameter tuning across LR, RF, XGBoost, CatBoost
6. Stacking ensemble with Platt-calibrated Random Forest
7. Threshold sweep to find the leaderboard-optimal cut-off
8. SHAP explainability on the best model

---

## Key Results

| Version | Val ROC-AUC | LB Error Rate |
|---------|------------|---------------|
| Baseline | 0.6885 | 0.4007 |
| Recency features | 0.6929 | 0.3448 |
| Time-based validation | 0.7214 | 0.2503 |
| PLTR features + CatBoost + F2 threshold | 0.7327 | **0.2076** |

**Best model:** LR + RF stacking ensemble, PLTR interaction features, threshold 0.25

**Key finding:** The single biggest improvement came from switching to a time-based validation split. The dataset covers only July 2017 (30 days). A random split inflated validation metrics and hid genuine model improvements for the first four iterations.

---

## Setup

```bash
conda env create -f environment.yml
conda activate loan-default-risk
```

---

## Techniques Demonstrated

- Multi-table data merging and behavioural feature engineering
- Imbalanced classification (SMOTE ablation, class weights, F2 threshold optimisation)
- Stacking ensembles with probability calibration
- PLTR interaction features for logistic regression
- Time-based validation for temporal datasets
- SHAP explainability
- Credit risk modelling concepts

---