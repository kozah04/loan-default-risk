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


def find_optimal_threshold(y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
    """
    Find the threshold that maximises the F1 score for the minority class (Bad loans).

    In a credit risk context, missing a bad loan (false negative) is typically
    more costly than incorrectly flagging a good loan (false positive). This
    function finds the threshold that best captures bad loans without being
    too aggressive.

    Parameters
    ----------
    y_true : pd.Series
    y_pred_proba : np.ndarray

    Returns
    -------
    float : optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
    return optimal_threshold


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
    Generate a SHAP summary plot for tree-based models.

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

    # Extract the underlying model from the pipeline if needed
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    explainer = shap.TreeExplainer(estimator)

    # Transform X through the pipeline up to but not including the model step
    if hasattr(model, "named_steps"):
        steps_before_model = list(model.named_steps.items())[:-1]
        X_transformed = X.copy()
        for _, step in steps_before_model:
            X_transformed = step.transform(X_transformed)
        X_transformed = pd.DataFrame(X_transformed, columns=X.columns)
    else:
        X_transformed = X

    shap_values = explainer.shap_values(X_transformed)

    # For binary classification, shap_values may be a list — take index 1 (positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()
