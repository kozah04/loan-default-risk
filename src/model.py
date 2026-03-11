"""
model.py
--------
Model training, hyperparameter tuning, and inference for the
SuperLender loan default prediction pipeline.

Three models are compared:
- Logistic Regression
- Random Forest
- XGBoost

Iteration 4 additions:
- run_smote_ablation(): controlled comparison of 4 imbalance strategies
  (SMOTE only, class_weight only, SMOTE+Tomek, none) per model
- calibrate_model(): Platt scaling for RF probability calibration
- build_stacking_ensemble(): trains LR/RF/XGB as base learners using
  out-of-fold predictions to avoid leakage, then fits a Logistic
  Regression meta-learner on those predictions
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, make_scorer, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
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


def get_models(class_weight: str = "balanced") -> dict:
    """
    Return a dictionary of model name to configured pipeline.

    Parameters
    ----------
    class_weight : str or None
        'balanced' applies class weights to LR and RF. Pass None
        to disable class weights (used in SMOTE-only ablation).

    Returns
    -------
    dict : {model_name: Pipeline}
    """
    models = {
        "logistic_regression": build_pipeline(
            LogisticRegression(
                class_weight=class_weight, max_iter=1000, random_state=42
            ),
            scale=True
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                class_weight=class_weight, random_state=42, n_jobs=-1
            )
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                scale_pos_weight=3,
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
    Parameters are prefixed with 'model__' to target the model step
    inside the Pipeline.

    XGBoost uses F1 Bad as its tuning metric (set in tune_model) and
    includes scale_pos_weight in the search space to let the tuner
    find the right minority class weighting.

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
            "model__scale_pos_weight": [1, 2, 3, 4, 5],
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

    XGBoost uses F1 for the Bad class as the scoring metric.
    All other models use ROC-AUC.

    Parameters
    ----------
    name : str
    pipeline : Pipeline
    param_grid : dict
    X_train : pd.DataFrame
    y_train : pd.Series
    n_iter : int
    cv : int

    Returns
    -------
    RandomizedSearchCV
    """
    print(f"Tuning {name}...")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    if "xgboost" in name.lower():
        scorer = make_scorer(f1_score, pos_label=0, zero_division=0)
    else:
        scorer = "roc_auc"

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  Best score: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search


def run_smote_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    Compare four imbalance-handling strategies across all three models.

    Strategies tested:
    - none: no resampling, no class weights
    - class_weight: class_weight='balanced', no resampling
    - smote: SMOTE oversampling, no class weights
    - smote_tomek: SMOTE + Tomek Links, no class weights

    For each strategy and model, trains with best known hyperparameters
    from previous iterations and evaluates on the validation set.
    The goal is to identify which strategy gives each model the best
    recall and F1 on the Bad class before spending compute on tuning.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val, y_val : validation features and labels

    Returns
    -------
    pd.DataFrame with columns: model, strategy, roc_auc, f1_bad, recall_bad
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_imp = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imp)

    strategies = {
        "none":         (None,                        None),
        "class_weight": (None,                        "balanced"),
        "smote":        (SMOTE(random_state=42),      None),
        "smote_tomek":  (SMOTETomek(random_state=42), None),
    }

    results = []

    for strategy_name, (resampler, cw) in strategies.items():
        print(f"  Running strategy: {strategy_name}")
        if resampler is not None:
            X_res, y_res = resampler.fit_resample(X_train_scaled, y_train)
        else:
            X_res, y_res = X_train_scaled, y_train

        model_configs = {
            "logistic_regression": LogisticRegression(
                class_weight=cw, max_iter=1000, random_state=42,
                C=0.001, solver="liblinear"
            ),
            "random_forest": RandomForestClassifier(
                class_weight=cw, random_state=42, n_jobs=-1,
                n_estimators=300, max_depth=5, min_samples_leaf=4,
                min_samples_split=10
            ),
            "xgboost": XGBClassifier(
                scale_pos_weight=3 if cw is None else 1,
                eval_metric="logloss", random_state=42, n_jobs=-1,
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.7, colsample_bytree=0.7
            ),
        }

        for model_name, model in model_configs.items():
            model.fit(X_res, y_res)
            proba = model.predict_proba(X_val_scaled)[:, 1]
            pred = (proba >= 0.5).astype(int)

            results.append({
                "model": model_name,
                "strategy": strategy_name,
                "roc_auc": round(roc_auc_score(y_val, proba), 4),
                "f1_bad": round(f1_score(y_val, pred, pos_label=0, zero_division=0), 4),
                "recall_bad": round(recall_score(y_val, pred, pos_label=0, zero_division=0), 4),
            })

    return pd.DataFrame(results).sort_values(
        ["model", "f1_bad"], ascending=[True, False]
    ).reset_index(drop=True)


def calibrate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """
    Apply Platt scaling (sigmoid) or isotonic regression to a fitted model.

    Random Forest probabilities tend to cluster near 0 and 1 rather than
    reflecting true class probabilities. Calibrating before stacking ensures
    the meta-learner receives meaningful estimates from all base learners.

    Parameters
    ----------
    model : fitted estimator or Pipeline
    X_train : pd.DataFrame
    y_train : pd.Series
    method : str
        'sigmoid' for Platt scaling (recommended for small data),
        'isotonic' for isotonic regression (needs more data to fit well)

    Returns
    -------
    CalibratedClassifierCV : fitted calibrated model
    """
    # cv="prefit" was removed in recent sklearn. cv=5 refits inside CV folds,
    # avoiding calibrating on the same data the model trained on.
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def build_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_models: dict,
) -> StackingClassifier:
    """
    Build a stacking ensemble using tuned LR, RF, and XGB as base learners
    and a Logistic Regression meta-learner.

    The base learners make out-of-fold predictions on the training data
    (handled internally by StackingClassifier with cv=5). These OOF
    predictions are used to train the meta-learner, which prevents
    leakage — the meta-learner never sees predictions made on data the
    base learners were trained on.

    RF is calibrated with Platt scaling before entering the stack so
    the meta-learner receives well-calibrated probability estimates from
    all three base learners.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    best_models : dict
        Dictionary of {model_name: fitted Pipeline} from the tuning step.

    Returns
    -------
    StackingClassifier : fitted stacking ensemble
    """
    lr_base  = best_models["logistic_regression"]
    rf_base  = best_models["random_forest"]
    xgb_base = best_models["xgboost"]

    print("Calibrating Random Forest probabilities (Platt scaling)...")
    rf_calibrated = calibrate_model(rf_base, X_train, y_train, method="sigmoid")

    estimators = [
    ("logistic_regression", lr_base),
    ("random_forest",       rf_calibrated),
    ]

    meta_learner = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42, C=0.1
    )

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    print("Fitting stacking ensemble...")
    stack.fit(X_train, y_train)
    return stack


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