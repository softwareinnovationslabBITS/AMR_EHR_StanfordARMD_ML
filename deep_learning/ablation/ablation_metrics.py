"""
Metric helpers for the TabTransformer ablation study.

#migrate: extracted from tabtransformer_ablation.py
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ExperimentMetrics:
    experiment: str
    removed_group: str
    n_removed_features: int
    n_retained_features: int
    n_cat_features: int
    n_cont_features: int
    n_binary_features: int
    best_epoch: int
    best_val_roc_auc: float
    validation_threshold: float
    test_accuracy: float
    test_precision: float
    test_recall_sensitivity: float
    test_specificity: float
    test_f1: float
    test_roc_auc: float
    test_pr_auc: float
    test_balanced_accuracy: float
    test_mcc: float
    test_cohen_kappa: float
    test_brier_score: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    training_minutes: float


def calculate_test_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float | int]:
    """Calculate all test-set performance metrics for an ablation experiment."""
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "test_accuracy": accuracy_score(labels, predictions),
        "test_precision": precision_score(labels, predictions, zero_division=0),
        "test_recall_sensitivity": recall_score(labels, predictions, zero_division=0),
        "test_specificity": specificity,
        "test_f1": f1_score(labels, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(labels, probabilities),
        "test_pr_auc": average_precision_score(labels, probabilities),
        "test_balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "test_mcc": matthews_corrcoef(labels, predictions),
        "test_cohen_kappa": cohen_kappa_score(labels, predictions),
        "test_brier_score": brier_score_loss(labels, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
