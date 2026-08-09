"""
Bootstrap sampling and confidence-interval logic for the baseline AMR
TabTransformer model.

#migrate: extracted from bootstrap_ci.py
"""

import logging
import time
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_stratified_bootstrap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    calculate_metrics,
    n_bootstraps: int,
    seed: int,
    progress_every: int = 250,
) -> pd.DataFrame:
    """
    Perform a stratified nonparametric bootstrap.

    The number of class-0 and class-1 records in each replicate is held equal
    to the corresponding count in the original test set. Labels and predicted
    probabilities are always sampled as paired observations.
    """
    rng = np.random.default_rng(seed)

    negative_indices = np.flatnonzero(y_true == 0)
    positive_indices = np.flatnonzero(y_true == 1)

    if len(negative_indices) == 0 or len(positive_indices) == 0:
        raise ValueError("The test set must contain both outcome classes.")

    rows = []
    started = time.time()

    # #migrate: log bootstrap progress every 250 iterations
    for iteration in range(1, n_bootstraps + 1):
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        sampled_indices = np.concatenate([sampled_negative, sampled_positive])
        # Shuffling is not mathematically required for the metrics but keeps
        # each replicate in a conventional random order.
        rng.shuffle(sampled_indices)

        replicate_metrics = calculate_metrics(
            y_true[sampled_indices],
            y_prob[sampled_indices],
            threshold,
        )
        replicate_metrics["bootstrap_iteration"] = iteration
        rows.append(replicate_metrics)

        if iteration % progress_every == 0 or iteration == n_bootstraps:
            elapsed_minutes = (time.time() - started) / 60.0
            logger.info(
                "Bootstrap %d/%d completed (%.2f minutes elapsed)",
                iteration,
                n_bootstraps,
                elapsed_minutes,
            )

    columns_first = ["bootstrap_iteration"]
    result = pd.DataFrame(rows)
    return result[columns_first + [c for c in result.columns if c not in columns_first]]


def summarize_bootstrap(
    point_estimates: Dict[str, float],
    bootstrap_df: pd.DataFrame,
    confidence_level: float,
    threshold: float,
    threshold_method: str,
) -> pd.DataFrame:
    """Create percentile confidence intervals and bootstrap standard errors."""
    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    metric_names = [
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "negative_predictive_value",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "brier_score",
    ]

    rows = []
    for metric in metric_names:
        values = bootstrap_df[metric].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]

        rows.append(
            {
                "metric": metric,
                "point_estimate": point_estimates[metric],
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_standard_error": float(np.std(values, ddof=1)),
                "ci_lower": float(np.percentile(values, lower_percentile)),
                "ci_upper": float(np.percentile(values, upper_percentile)),
                "confidence_level": confidence_level,
                "n_valid_replicates": int(len(values)),
                "bootstrap_method": "stratified_percentile",
                "fixed_threshold": threshold,
                "threshold_selected_on": "validation_set",
                "threshold_selection_method": threshold_method,
            }
        )

    return pd.DataFrame(rows)
