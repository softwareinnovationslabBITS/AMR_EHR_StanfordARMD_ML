"""
Forest plots and visualization for bootstrap confidence intervals of the
baseline AMR TabTransformer model.

#migrate: extracted from bootstrap_ci.py
"""

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall_sensitivity": "Recall / sensitivity",
    "specificity": "Specificity",
    "negative_predictive_value": "Negative predictive value",
    "f1_score": "F1-score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "balanced_accuracy": "Balanced accuracy",
    "mcc": "Matthews correlation coefficient",
    "cohen_kappa": "Cohen's kappa",
    "brier_score": "Brier score",
}


def create_manuscript_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table with manuscript-ready estimate (95% CI) text."""
    manuscript = summary_df.copy()
    manuscript["metric_display"] = manuscript["metric"].map(METRIC_DISPLAY_NAMES)
    manuscript["estimate_95_ci"] = manuscript.apply(
        lambda row: (
            f"{row['point_estimate']:.4f} "
            f"({row['ci_lower']:.4f}-{row['ci_upper']:.4f})"
        ),
        axis=1,
    )
    return manuscript[
        [
            "metric_display",
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "estimate_95_ci",
            "bootstrap_standard_error",
            "n_valid_replicates",
            "fixed_threshold",
        ]
    ]


def make_forest_plot(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a horizontal confidence-interval plot for all metrics."""
    plot_df = summary_df.copy()
    plot_df["metric_display"] = plot_df["metric"].map(METRIC_DISPLAY_NAMES)

    # Brier score has the opposite interpretation, but it can still be shown.
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    y_position = np.arange(len(plot_df))

    lower_error = plot_df["point_estimate"] - plot_df["ci_lower"]
    upper_error = plot_df["ci_upper"] - plot_df["point_estimate"]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.errorbar(
        plot_df["point_estimate"],
        y_position,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        capsize=3,
        linewidth=1.4,
    )
    ax.set_yticks(y_position)
    ax.set_yticklabels(plot_df["metric_display"])
    ax.set_xlabel("Point estimate and percentile 95% confidence interval")
    ax.set_title("Baseline TabTransformer performance with bootstrap confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        output_dir / "baseline_ci_forest_plot_v2.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "baseline_ci_forest_plot_v2.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_distribution_plot(
    bootstrap_df: pd.DataFrame,
    point_estimates: Dict[str, float],
    output_dir: Path,
) -> None:
    """Plot distributions for the principal manuscript metrics."""
    selected_metrics = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "f1_score",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for axis, metric in zip(axes, selected_metrics):
        axis.hist(bootstrap_df[metric].dropna(), bins=40, alpha=0.85)
        axis.axvline(
            point_estimates[metric],
            linestyle="--",
            linewidth=1.5,
            label="Point estimate",
        )
        axis.set_title(METRIC_DISPLAY_NAMES[metric])
        axis.set_xlabel("Bootstrap value")
        axis.set_ylabel("Frequency")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)

    fig.suptitle(
        "Bootstrap distributions for principal baseline-model metrics",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "baseline_bootstrap_distributions_v2.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "baseline_bootstrap_distributions_v2.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
