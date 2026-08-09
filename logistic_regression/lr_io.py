"""
Save/load helpers for metrics, predictions, model, and config for the
feature-matched logistic regression benchmark.

#migrate: extracted from logistic_regression_dl_matched.py
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_test_metrics_row(
    metrics_row: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save the single-row test metrics CSV."""
    pd.DataFrame([metrics_row]).to_csv(
        output_dir / "logistic_regression_test_metrics.csv",
        index=False,
    )
    logger.info("Saved test metrics CSV")


def save_threshold_comparison(
    test_metrics: Dict[str, float | int],
    default_metrics: Dict[str, float | int],
    output_dir: Path,
) -> None:
    """Save the threshold comparison CSV."""
    pd.DataFrame(
        [
            {"threshold_type": "validation_MCC_optimized", **test_metrics},
            {"threshold_type": "default_0.50", **default_metrics},
        ]
    ).to_csv(output_dir / "threshold_comparison.csv", index=False)
    logger.info("Saved threshold comparison CSV")


def save_predictions(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    best_threshold: float,
    output_dir: Path,
) -> None:
    """Save test-set predictions."""
    prediction_df = pd.DataFrame(
        {
            "y_true": y_test,
            "probability_resistant": test_prob,
            "predicted_class": (test_prob >= best_threshold).astype(np.int8),
        }
    )
    prediction_df.to_csv(output_dir / "test_predictions.csv", index=False)
    logger.info("Saved test predictions CSV")


def save_bootstrap_results(
    bootstrap_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save bootstrap confidence intervals."""
    bootstrap_df.to_csv(
        output_dir / "logistic_regression_bootstrap_95CI.csv",
        index=False,
    )
    logger.info("Saved bootstrap confidence intervals")


def save_model_and_encoder(
    model,
    encoder,
    output_dir: Path,
) -> None:
    """Persist the fitted logistic regression model and one-hot encoder."""
    joblib.dump(model, output_dir / "logistic_regression_model.joblib")
    joblib.dump(encoder, output_dir / "onehot_encoder.joblib")
    logger.info("Saved model and encoder")


def save_run_config(
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist the run configuration as JSON."""
    with open(output_dir / "run_config.json", "w") as handle:
        json.dump(config, handle, indent=2)
    logger.info("Saved run config")
