"""
Metric calculation, threshold selection, and bootstrap confidence intervals
for the feature-matched logistic regression benchmark.

#migrate: extracted from logistic_regression_dl_matched.py
"""

import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd
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

logger = logging.getLogger(__name__)


def select_threshold_by_mcc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_threshold_candidates: int = 1001,
) -> tuple:
    """Select the classification threshold that maximizes validation MCC."""
    thresholds = np.linspace(0.0, 1.0, n_threshold_candidates)

    best_threshold = 0.5
    best_mcc = -np.inf

    rows = []

    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(np.int8)

        # Avoid degenerate one-class predictions causing meaningless values.
        if len(np.unique(pred)) < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(y_true, pred)

        rows.append({"threshold": threshold, "mcc": mcc})

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    threshold_df = pd.DataFrame(rows)

    return best_threshold, best_mcc, threshold_df


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float | int]:
    """Calculate the full set of performance metrics for a given threshold."""
    pred = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "npv": npv,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "cohen_kappa": cohen_kappa_score(y_true, pred),
        "brier_score": brier_score_loss(y_true, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample class-stratified bootstrap indices."""
    idx_0 = np.flatnonzero(y == 0)
    idx_1 = np.flatnonzero(y == 1)

    boot_0 = rng.choice(idx_0, size=len(idx_0), replace=True)
    boot_1 = rng.choice(idx_1, size=len(idx_1), replace=True)

    idx = np.concatenate([boot_0, boot_1])
    rng.shuffle(idx)

    return idx


def run_stratified_bootstrap(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    best_threshold: float,
    n_bootstraps: int = 2000,
    bootstrap_seed: int = 42,
    progress_every: int = 250,
) -> Dict[str, List[float]]:
    """
    Perform stratified nonparametric bootstrap resampling of paired test
    labels and probabilities.
    """
    bootstrap_metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "npv",
        "f1",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "brier_score",
    ]

    bootstrap_results = {metric: [] for metric in bootstrap_metrics}
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_start = time.time()

    # #migrate: log bootstrap progress every 250 iterations
    for b in range(n_bootstraps):
        idx = _stratified_bootstrap_indices(y_test, rng)

        y_b = y_test[idx]
        p_b = test_prob[idx]

        m = calculate_metrics(y_b, p_b, best_threshold)

        for metric in bootstrap_metrics:
            bootstrap_results[metric].append(m[metric])

        if (b + 1) % progress_every == 0 or b == 0:
            elapsed_minutes = (time.time() - bootstrap_start) / 60.0
            logger.info(
                "Bootstrap %d/%d completed (%.2f minutes elapsed)",
                b + 1,
                n_bootstraps,
                elapsed_minutes,
            )

    return bootstrap_results


def summarize_bootstrap_ci(
    bootstrap_results: Dict[str, List[float]],
    test_metrics: Dict[str, float | int],
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Create percentile confidence intervals from bootstrap replicates."""
    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    bootstrap_rows = []

    for metric, values in bootstrap_results.items():
        values_arr = np.asarray(values, dtype=np.float64)
        point_estimate = test_metrics[metric]
        lower = np.percentile(values_arr, lower_percentile)
        upper = np.percentile(values_arr, upper_percentile)

        bootstrap_rows.append(
            {
                "metric": metric,
                "estimate": point_estimate,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
            }
        )

    return pd.DataFrame(bootstrap_rows)
