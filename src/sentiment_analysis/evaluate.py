"""Evaluation helpers for classification models."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DEFAULT_TARGET_NAMES = ["negative", "neutral", "positive"]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = "Test",
    target_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Print and return accuracy and classification report metrics."""
    target_names = target_names or DEFAULT_TARGET_NAMES
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4
    )
    matrix = confusion_matrix(y_true, y_pred)

    print(f"\n--- {split_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    return {
        "accuracy": float(acc),
        "report": report,
        "confusion_matrix": matrix,
    }


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "Test",
    target_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Predict on X and evaluate against y."""
    y_pred = model.predict(X)
    return evaluate_predictions(
        y_true=y,
        y_pred=y_pred,
        split_name=split_name,
        target_names=target_names,
    )

