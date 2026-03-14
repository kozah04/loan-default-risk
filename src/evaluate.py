"""
evaluate.py
-----------
Evaluation utilities for the SuperLender loan default pipeline.

Covers:
- Classification metrics (F1, ROC-AUC, precision, recall)
- Confusion matrix plotting
- ROC curve plotting
- Threshold optimisation — finding the decision boundary that best
  balances precision and recall given the business context
- SHAP summary plots for model explainability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_recall_curve,
)
from pathlib import Path


FIGURES_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def evaluate_model(y_true: pd.Series, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics at a given threshold.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold. Default is 0.5.

    Returns
    -------
    dict of metric name to value
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_bad": report["0"]["precision"],
        "recall_bad": report["0"]["recall"],
        "f1_bad": report["0"]["f1-score"],
        "precision_good": report["1"]["precision"],
        "recall_good": report["1"]["recall"],
        "f1_good": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
        "threshold": threshold,
    }


def find_optimal_threshold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    beta: float = 2.0,
) -> float:
    """
    Find the threshold that maximises Fbeta for the Bad class (class 0).

    predict_proba returns the probability of Good (class 1). A customer is
    predicted Bad when their Good probability falls below the threshold.

    We use F2 (beta=2) by default rather than F1. F2 weights recall twice
    as heavily as precision, which aligns with two observations from this
    project:
    - Missing a defaulter (false negative) is more costly than a false alarm
    - The leaderboard-optimal threshold (0.39) is consistently lower than
      the F1-optimal threshold (0.47), meaning the leaderboard rewards
      recall more aggressively than F1 captures

    Research on imbalanced credit classification confirms that optimising
    Fbeta with beta > 1 naturally finds lower thresholds that better match
    recall-weighted business objectives (irjmets.com, 2024).

    Parameters
    ----------
    y_true : pd.Series
        True binary labels where 1 = Good and 0 = Bad.
    y_pred_proba : np.ndarray
        Predicted probabilities for the Good class (class 1).
    beta : float
        Beta parameter for Fbeta score. Default 2.0 weights recall
        twice as heavily as precision. Use 1.0 for standard F1.

    Returns
    -------
    float : optimal threshold
    """
    from sklearn.metrics import fbeta_score

    thresholds = np.arange(0.25, 0.65, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, pos_label=0, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"Optimal threshold: {best_threshold:.4f} (F{beta} Bad class: {best_score:.4f})")
    return best_threshold


def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True) -> None:
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Bad", "Predicted Good"],
        yticklabels=["Actual Bad", "Actual Good"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.show()


def plot_roc_curves(results: dict, save: bool = True) -> None:
    """
    Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    results : dict
        {model_name: {"y_true": ..., "y_pred_proba": ...}}
    save : bool
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_pred_proba"])
        auc = roc_auc_score(data["y_true"], data["y_pred_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "roc_curves_comparison.png", dpi=150)
    plt.show()


def plot_shap_summary(model, X: pd.DataFrame, model_name: str, save: bool = True) -> None:
    """
    Generate a SHAP summary plot. Automatically selects the right explainer
    based on model type: TreeExplainer for tree-based models (Random Forest,
    XGBoost) and LinearExplainer for linear models (Logistic Regression).

    Parameters
    ----------
    model : fitted sklearn Pipeline or estimator
    X : pd.DataFrame
        Feature matrix (unscaled, pre-pipeline).
    model_name : str
    save : bool
    """
    try:
        import shap
    except ImportError:
        print("shap is not installed. Run: pip install shap")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # Extract the underlying model from the pipeline
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    # Transform X through all pipeline steps except the final model step
    if hasattr(model, "named_steps"):
        steps_before_model = list(model.named_steps.items())[:-1]
        X_transformed = X.copy()
        for _, step in steps_before_model:
            X_transformed = step.transform(X_transformed)
        X_transformed = pd.DataFrame(X_transformed, columns=X.columns)
    else:
        X_transformed = X

    # Select the right explainer based on model type
    if isinstance(estimator, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)
        # For Random Forest binary classification, shap_values is a list — take index 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif isinstance(estimator, LogisticRegression):
        explainer = shap.LinearExplainer(estimator, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        # Fall back to KernelExplainer for any other model type — slower but universal
        print(f"Using KernelExplainer for {type(estimator).__name__} (this may be slow).")
        explainer = shap.KernelExplainer(estimator.predict_proba, shap.sample(X_transformed, 100))
        shap_values = explainer.shap_values(X_transformed)[1]

    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()