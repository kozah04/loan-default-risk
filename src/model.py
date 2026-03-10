"""
model.py
--------
Model training, hyperparameter tuning, and inference for the
SuperLender loan default prediction pipeline.

Three models are compared:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Each model is tuned using RandomizedSearchCV. Final model selection
is based on ROC-AUC and F1 score on the validation set, with class
imbalance handled via class_weight and SMOTE.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"


def build_pipeline(model, scale: bool = False) -> Pipeline:
    """
    Wrap a model in a sklearn Pipeline with imputation and optional scaling.

    Imputation uses median strategy so missing behavioural features
    (for the very small number of customers with no prior history)
    are filled with sensible values rather than dropped.

    Parameters
    ----------
    model : sklearn-compatible estimator
    scale : bool
        Whether to apply StandardScaler. Required for Logistic Regression,
        not needed for tree-based models.

    Returns
    -------
    sklearn Pipeline
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_models() -> dict:
    """
    Return a dictionary of model name to configured pipeline.
    Class weights are set to 'balanced' for Logistic Regression and
    Random Forest to handle the 78/22 class imbalance without SMOTE.
    XGBoost uses scale_pos_weight instead.

    Returns
    -------
    dict : {model_name: Pipeline}
    """
    models = {
        "logistic_regression": build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            scale=True
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                scale_pos_weight=3,  # approx ratio of majority/minority class
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        ),
    }
    return models


def get_param_grids() -> dict:
    """
    Return hyperparameter search grids for each model.
    Parameters are prefixed with 'model__' to target the
    model step inside the Pipeline.

    Returns
    -------
    dict : {model_name: param_grid}
    """
    param_grids = {
        "logistic_regression": {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "xgboost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        },
    }
    return param_grids


def tune_model(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
) -> RandomizedSearchCV:
    """
    Run RandomizedSearchCV for a given model pipeline.

    Parameters
    ----------
    name : str
        Model name for logging.
    pipeline : Pipeline
        Model pipeline to tune.
    param_grid : dict
        Hyperparameter search space.
    X_train : pd.DataFrame
    y_train : pd.Series
    n_iter : int
        Number of parameter combinations to try.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object with best estimator available via .best_estimator_
    """
    print(f"Tuning {name}...")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  Best ROC-AUC: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search


def save_model(model, name: str) -> None:
    """Save a fitted model to outputs/models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(name: str):
    """Load a saved model from outputs/models/."""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probabilities for the positive class."""
    return model.predict_proba(X)[:, 1]
