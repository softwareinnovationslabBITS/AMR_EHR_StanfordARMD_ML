"""
Matplotlib plotting helpers for the feature-matched logistic regression
benchmark.

#migrate: extracted from logistic_regression_dl_matched.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


def plot_roc_curve(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    test_metrics: Dict[str, float],
    output_dir: Path,
) -> None:
    """Save the ROC curve for the test set."""
    fpr, tpr, _ = roc_curve(y_test, test_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"Logistic regression "
            f"(AUC={test_metrics['roc_auc']:.3f})"
        ),
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curve")


def plot_precision_recall_curve(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    test_metrics: Dict[str, float],
    output_dir: Path,
) -> None:
    """Save the precision-recall curve for the test set."""
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_prob)
    prevalence = np.mean(y_test)

    plt.figure(figsize=(7, 6))
    plt.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=(
            f"Logistic regression "
            f"(AP={test_metrics['pr_auc']:.3f})"
        ),
    )
    plt.axhline(
        prevalence,
        linestyle="--",
        label=f"Resistance prevalence={prevalence:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Logistic Regression Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved precision-recall curve")


def plot_confusion_matrix(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    best_threshold: float,
    output_dir: Path,
) -> None:
    """Save the confusion matrix for the test set."""
    test_pred = (test_prob >= best_threshold).astype(np.int8)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Susceptible", "Resistant"])
    plt.yticks([0, 1], ["Susceptible", "Resistant"])
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(
        "Logistic Regression Confusion Matrix\n"
        f"Validation-selected threshold = {best_threshold:.3f}"
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrix")


def plot_calibration_curve(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    n_calibration_bins: int,
    output_dir: Path,
) -> None:
    """Save the calibration curve for the test set."""
    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        test_prob,
        n_bins=n_calibration_bins,
        strategy="quantile",
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        linewidth=2,
        label="Logistic regression",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed resistant proportion")
    plt.title("Logistic Regression Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "calibration_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved calibration curve")


def plot_validation_threshold_curve(
    threshold_df: pd.DataFrame,
    best_threshold: float,
    output_dir: Path,
) -> None:
    """Save the validation MCC versus threshold curve."""
    plt.figure(figsize=(7, 6))
    plt.plot(threshold_df["threshold"], threshold_df["mcc"])
    plt.axvline(
        best_threshold,
        linestyle="--",
        label=(f"Selected threshold={best_threshold:.3f}"),
    )
    plt.xlabel("Classification threshold")
    plt.ylabel("Validation MCC")
    plt.title("Validation Threshold Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "validation_threshold_mcc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved validation threshold curve")


def plot_coefficient_importance(
    model,
    expanded_feature_names: Sequence[str],
    top_n: int,
    output_dir: Path,
) -> None:
    """Save the top-N coefficient importance plot."""
    coefficients = model.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": expanded_feature_names,
            "coefficient": coefficients,
        }
    )
    coef_df["absolute_coefficient"] = np.abs(coef_df["coefficient"])
    coef_df = coef_df.sort_values("absolute_coefficient", ascending=False)
    coef_df.to_csv(output_dir / "all_logistic_coefficients.csv", index=False)

    top_coef = coef_df.head(top_n).copy()
    top_coef = top_coef.iloc[::-1]

    plt.figure(figsize=(10, 10))
    plt.barh(top_coef["feature"], top_coef["coefficient"])
    plt.xlabel("Logistic regression coefficient")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig(
        output_dir / "top_logistic_coefficients.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    logger.info("Saved coefficient importance plot")


def generate_all_plots(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    test_metrics: Dict[str, float],
    threshold_df: pd.DataFrame,
    best_threshold: float,
    model,
    expanded_feature_names: Sequence[str],
    n_calibration_bins: int,
    top_n_coefficients: int,
    output_dir: Path,
) -> None:
    """Generate and save all logistic regression figures."""
    logger.info("Generating plots")
    plot_roc_curve(y_test, test_prob, test_metrics, output_dir)
    plot_precision_recall_curve(y_test, test_prob, test_metrics, output_dir)
    plot_confusion_matrix(y_test, test_prob, best_threshold, output_dir)
    plot_calibration_curve(y_test, test_prob, n_calibration_bins, output_dir)
    plot_validation_threshold_curve(threshold_df, best_threshold, output_dir)
    plot_coefficient_importance(
        model,
        expanded_feature_names,
        top_n_coefficients,
        output_dir,
    )
    logger.info("All plots saved")
