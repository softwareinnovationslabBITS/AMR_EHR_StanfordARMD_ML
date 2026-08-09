"""
Plotting helpers for the baseline TabTransformer final loss evaluation.

#migrate: extracted from tabtransformer_loss_evaluation.py
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_and_final_split_losses(
    training_loss_history: np.ndarray,
    validation_auc_history: Optional[np.ndarray],
    final_train_loss: float,
    final_validation_loss: float,
    final_test_loss: float,
    output_dir: Path,
) -> None:
    """Save the per-epoch training loss plus final split-loss reference lines."""
    epochs = np.arange(1, len(training_loss_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        training_loss_history,
        linewidth=2.2,
        label="Training loss recorded during training",
    )

    ax.axhline(
        final_train_loss,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Final saved-model training loss "
            f"= {final_train_loss:.4f}"
        ),
    )

    ax.axhline(
        final_validation_loss,
        linestyle=":",
        linewidth=1.8,
        label=(
            "Final saved-model validation loss "
            f"= {final_validation_loss:.4f}"
        ),
    )

    ax.axhline(
        final_test_loss,
        linestyle="-.",
        linewidth=1.8,
        label=(
            "Final saved-model test loss "
            f"= {final_test_loss:.4f}"
        ),
    )

    if (
        validation_auc_history is not None
        and len(validation_auc_history) == len(training_loss_history)
    ):
        best_epoch = int(
            np.argmax(validation_auc_history)
            + 1
        )

        best_validation_auc = float(
            np.max(validation_auc_history)
        )

        ax.axvline(
            best_epoch,
            linestyle="--",
            linewidth=1.2,
            label=(
                f"Best saved epoch = {best_epoch} "
                f"(validation ROC-AUC "
                f"= {best_validation_auc:.4f})"
            ),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Weighted binary cross-entropy loss"
    )

    ax.set_title(
        "Baseline TabTransformer training loss and "
        "final split losses"
    )

    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "baseline_training_and_final_split_losses_v3.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "baseline_training_and_final_split_losses_v3.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_final_loss_comparison(
    final_train_loss: float,
    final_validation_loss: float,
    final_test_loss: float,
    output_dir: Path,
) -> None:
    """Save the final train-validation-test loss bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    split_names = [
        "Training",
        "Validation",
        "Test",
    ]

    split_losses = [
        final_train_loss,
        final_validation_loss,
        final_test_loss,
    ]

    bars = ax.bar(
        split_names,
        split_losses,
    )

    ax.set_ylabel(
        "Weighted binary cross-entropy loss"
    )

    ax.set_title(
        "Final loss of the saved baseline TabTransformer"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        split_losses,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    fig.savefig(
        output_dir / "baseline_final_train_validation_test_loss_v3.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "baseline_final_train_validation_test_loss_v3.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)
