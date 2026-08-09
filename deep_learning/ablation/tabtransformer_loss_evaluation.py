#!/usr/bin/env python3
# Source: /AMR_Stanford/DL_codes/amr_project/amr_baseline_tabtransformer_loss_evaluation.py
"""
Standalone loss evaluation and visualization for the saved baseline
AMR TabTransformer model.

What this script can produce
----------------------------
1. The saved training loss for every training epoch.
2. The loss of the final/best saved model on:
   - training set
   - validation set
   - test set
3. A figure containing:
   - per-epoch training loss
   - horizontal reference lines for final train, validation, and test loss
4. A bar chart comparing final train, validation, and test loss.
5. CSV and JSON files containing the calculated values.

Important limitation
--------------------
The original training bundle saved per-epoch training loss and validation
ROC-AUC, but it did not save validation or test loss at every epoch, nor did
it save model weights for every epoch. Therefore, genuine validation/test
loss curves across all epochs cannot be reconstructed retrospectively.

Input files
-----------
amr_model.pt
amr_analysis_bundle.joblib

Output directory
----------------
baseline_final_loss_evaluation_v3/
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

# #migrate: split data loading/processing and plotting into private modules
from ablation_loss_data import AMRDataset, evaluate_mean_loss
from ablation_loss_plots import (
    plot_final_loss_comparison,
    plot_training_and_final_split_losses,
)

# #migrate: configure timestamped progress logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# #migrate: artifact paths updated from the single config file
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
_TT_CFG = _CFG.get('tabtransformer', {})
MODEL_PATH = resolve_path(_TT_CFG.get('model_path', 'models/tabtransformer/amr_model.pt'))
ANALYSIS_BUNDLE_PATH = resolve_path(_TT_CFG.get('bundle_path', 'dataset/amr_analysis_bundle.joblib'))

# #migrate: output directory from the single config file
OUTPUT_DIR = Path(str(resolve_path(_TT_CFG.get('final_loss_output_dir', 'deep_learning/baseline_final_loss_evaluation_v3'))))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# #migrate: batch size and seed from the single config file
BATCH_SIZE = _TT_CFG.get('batch_size', 512)
NUM_WORKERS = 0
RANDOM_SEED = _CFG.get('seed', 42)

# Use the same weighted BCE loss formulation as the original training script.
USE_CLASS_WEIGHTED_LOSS = True


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# =============================================================================
# 3. MODEL ARCHITECTURE
#    This must match the original baseline training architecture exactly.
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.attn(x, x, x)

        return self.norm(
            x + self.dropout(attention_output)
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)

        return self.norm(
            x + self.dropout(self.ff(x))
        )


class AMRTabTransformer(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Dict[str, int],
        cat_embed_dims: Dict[str, int],
        n_cont: int,
        n_bin: int,
        attn_embed_dim: int = 64,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        ff_dim: int = 256,
        mlp_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.cat_feature_names = list(cat_cardinalities.keys())

        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(
                        cat_cardinalities[feature],
                        cat_embed_dims[feature],
                    ),
                    nn.Linear(
                        cat_embed_dims[feature],
                        attn_embed_dim,
                    ),
                )
                for feature in self.cat_feature_names
            ]
        )

        self.transformer_layers = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=attn_embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.cont_bn = nn.BatchNorm1d(n_cont)

        self.cont_proj = nn.Sequential(
            nn.Linear(n_cont, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )

        self.bin_proj = nn.Sequential(
            nn.Linear(n_bin, 64) if n_bin > 0 else nn.Identity(),
            nn.GELU() if n_bin > 0 else nn.Identity(),
        )

        self.n_bin = n_bin

        self.wide_proj = nn.Linear(
            n_cont + n_bin,
            64,
        )

        n_cat = len(cat_cardinalities) * attn_embed_dim
        n_deep = 128 + (64 if n_bin > 0 else 0)
        total_input_dim = n_cat + n_deep + 64

        mlp_layers: List[nn.Module] = []
        input_dim = total_input_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        mlp_layers.append(
            nn.Linear(input_dim, 1)
        )

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        x_bin: torch.Tensor,
    ) -> torch.Tensor:

        categorical_embeddings = [
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]

        categorical_sequence = torch.stack(
            categorical_embeddings,
            dim=1,
        )

        categorical_sequence = self.transformer_layers(
            categorical_sequence
        )

        categorical_flat = categorical_sequence.flatten(1)

        continuous_normalized = self.cont_bn(x_cont)
        continuous_output = self.cont_proj(
            continuous_normalized
        )

        if self.n_bin > 0:
            binary_output = self.bin_proj(x_bin)

            deep_output = torch.cat(
                [continuous_output, binary_output],
                dim=1,
            )
        else:
            deep_output = continuous_output

        wide_input = torch.cat(
            [continuous_normalized, x_bin],
            dim=1,
        )

        wide_output = self.wide_proj(wide_input)

        fused_output = torch.cat(
            [
                categorical_flat,
                deep_output,
                wide_output,
            ],
            dim=1,
        )

        return self.mlp(fused_output).squeeze(1)


# =============================================================================
# 4-5. DATASET AND LOSS EVALUATION
#    #migrate: AMRDataset and evaluate_mean_loss moved to ablation_loss_data.py
# =============================================================================


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:

    logger.info("=" * 78)
    logger.info("BASELINE TABTRANSFORMER FINAL LOSS EVALUATION")
    logger.info("=" * 78)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.resolve()}"
        )

    if not ANALYSIS_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Analysis bundle not found: "
            f"{ANALYSIS_BUNDLE_PATH.resolve()}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Device: %s", device)

    if device.type == "cuda":
        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

    logger.info("Loading saved artifacts...")

    model_bundle = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    analysis_bundle = joblib.load(
        ANALYSIS_BUNDLE_PATH
    )

    required_model_keys = {
        "model_state_dict",
        "cat_cardinalities",
        "cat_embed_dims",
        "n_cont",
        "n_bin",
    }

    missing_model_keys = (
        required_model_keys
        - set(model_bundle.keys())
    )

    if missing_model_keys:
        raise KeyError(
            "Missing keys in model bundle: "
            + ", ".join(sorted(missing_model_keys))
        )

    required_analysis_keys = {
        "CAT_FEATURES",
        "CONT_FEATURES",
        "BINARY_FEATURES",
        "ALL_FEATURES",
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test",
    }

    missing_analysis_keys = (
        required_analysis_keys
        - set(analysis_bundle.keys())
    )

    if missing_analysis_keys:
        raise KeyError(
            "Missing keys in analysis bundle: "
            + ", ".join(sorted(missing_analysis_keys))
        )

    CAT_FEATURES = analysis_bundle["CAT_FEATURES"]
    CONT_FEATURES = analysis_bundle["CONT_FEATURES"]
    BINARY_FEATURES = analysis_bundle["BINARY_FEATURES"]
    ALL_FEATURES = analysis_bundle["ALL_FEATURES"]

    X_train = np.asarray(
        analysis_bundle["X_train"],
        dtype=np.float32,
    )

    X_val = np.asarray(
        analysis_bundle["X_val"],
        dtype=np.float32,
    )

    X_test = np.asarray(
        analysis_bundle["X_test"],
        dtype=np.float32,
    )

    y_train = np.asarray(
        analysis_bundle["y_train"],
        dtype=np.float32,
    ).ravel()

    y_val = np.asarray(
        analysis_bundle["y_val"],
        dtype=np.float32,
    ).ravel()

    y_test = np.asarray(
        analysis_bundle["y_test"],
        dtype=np.float32,
    ).ravel()

    cat_idx = list(
        range(len(CAT_FEATURES))
    )

    cont_idx = list(
        range(
            len(CAT_FEATURES),
            len(CAT_FEATURES)
            + len(CONT_FEATURES),
        )
    )

    bin_idx = list(
        range(
            len(CAT_FEATURES)
            + len(CONT_FEATURES),
            len(ALL_FEATURES),
        )
    )

    logger.info("Split sizes:")
    logger.info("  Train:      %s", f"{len(y_train):,}")
    logger.info("  Validation: %s", f"{len(y_val):,}")
    logger.info("  Test:       %s", f"{len(y_test):,}")

    model = AMRTabTransformer(
        cat_cardinalities=model_bundle[
            "cat_cardinalities"
        ],
        cat_embed_dims=model_bundle[
            "cat_embed_dims"
        ],
        n_cont=int(model_bundle["n_cont"]),
        n_bin=int(model_bundle["n_bin"]),
    )

    model.load_state_dict(
        model_bundle["model_state_dict"]
    )

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully.")

    # -------------------------------------------------------------------------
    # Recreate the original weighted BCE loss.
    # -------------------------------------------------------------------------

    if USE_CLASS_WEIGHTED_LOSS:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=y_train.astype(int),
        )

        positive_weight_value = (
            class_weights[1]
            / class_weights[0]
        )

        positive_weight = torch.tensor(
            positive_weight_value,
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight
        )

        logger.info(
            "Loss: weighted BCEWithLogitsLoss"
        )

        logger.info(
            "Positive-class weight: %.6f",
            positive_weight_value,
        )

    else:
        positive_weight_value = None
        criterion = nn.BCEWithLogitsLoss()

        logger.info(
            "Loss: unweighted BCEWithLogitsLoss"
        )

    # -------------------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------------------

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        AMRDataset(
            X_train,
            y_train,
            cat_idx,
            cont_idx,
            bin_idx,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        AMRDataset(
            X_val,
            y_val,
            cat_idx,
            cont_idx,
            bin_idx,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        AMRDataset(
            X_test,
            y_test,
            cat_idx,
            cont_idx,
            bin_idx,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # -------------------------------------------------------------------------
    # Final losses for the saved best model
    # -------------------------------------------------------------------------

    logger.info("Calculating final training loss...")
    final_train_loss = evaluate_mean_loss(
        model=model,
        loader=train_loader,
        criterion=criterion,
        device=device,
        split_name="Train",
    )

    logger.info("Calculating final validation loss...")
    final_validation_loss = evaluate_mean_loss(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        split_name="Validation",
    )

    logger.info("Calculating final test loss...")
    final_test_loss = evaluate_mean_loss(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split_name="Test",
    )

    logger.info("Final saved-model losses:")
    logger.info(
        "  Training loss:   %.6f",
        final_train_loss,
    )
    logger.info(
        "  Validation loss: %.6f",
        final_validation_loss,
    )
    logger.info(
        "  Test loss:       %.6f",
        final_test_loss,
    )

    # -------------------------------------------------------------------------
    # Retrieve the original saved per-epoch history.
    # -------------------------------------------------------------------------

    history = model_bundle.get(
        "history",
        None,
    )

    training_loss_history = None
    validation_auc_history = None

    if isinstance(history, dict):
        if "train_loss" in history:
            training_loss_history = np.asarray(
                history["train_loss"],
                dtype=float,
            )

        if "val_auc" in history:
            validation_auc_history = np.asarray(
                history["val_auc"],
                dtype=float,
            )

    if (
        training_loss_history is None
        or len(training_loss_history) == 0
    ):
        logger.warning(
            "No per-epoch training loss was found in the saved model."
        )

    # -------------------------------------------------------------------------
    # Save final loss table.
    # -------------------------------------------------------------------------

    final_loss_table = pd.DataFrame(
        {
            "dataset_split": [
                "Training",
                "Validation",
                "Test",
            ],
            "n_observations": [
                len(y_train),
                len(y_val),
                len(y_test),
            ],
            "positive_prevalence": [
                float(y_train.mean()),
                float(y_val.mean()),
                float(y_test.mean()),
            ],
            "weighted_binary_cross_entropy_loss": [
                final_train_loss,
                final_validation_loss,
                final_test_loss,
            ],
        }
    )

    final_loss_csv = (
        OUTPUT_DIR
        / "baseline_final_split_losses_v3.csv"
    )

    final_loss_table.to_csv(
        final_loss_csv,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save history table.
    # -------------------------------------------------------------------------

    if (
        training_loss_history is not None
        and len(training_loss_history) > 0
    ):
        history_table = pd.DataFrame(
            {
                "epoch": np.arange(
                    1,
                    len(training_loss_history) + 1,
                ),
                "training_loss": (
                    training_loss_history
                ),
            }
        )

        if (
            validation_auc_history is not None
            and len(validation_auc_history)
            == len(training_loss_history)
        ):
            history_table[
                "validation_roc_auc"
            ] = validation_auc_history

        history_csv = (
            OUTPUT_DIR
            / "baseline_saved_training_history_v3.csv"
        )

        history_table.to_csv(
            history_csv,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Plot 1: saved training-loss curve plus final split-loss references.
    # #migrate: plotting moved to ablation_loss_plots.py
    # -------------------------------------------------------------------------

    if (
        training_loss_history is not None
        and len(training_loss_history) > 0
    ):
        logger.info("Plotting training loss with final split-loss references")

        plot_training_and_final_split_losses(
            training_loss_history=training_loss_history,
            validation_auc_history=validation_auc_history,
            final_train_loss=final_train_loss,
            final_validation_loss=final_validation_loss,
            final_test_loss=final_test_loss,
            output_dir=OUTPUT_DIR,
        )

        logger.info("Training/split-loss plot saved")

    # -------------------------------------------------------------------------
    # Plot 2: final train-validation-test loss comparison.
    # #migrate: plotting moved to ablation_loss_plots.py
    # -------------------------------------------------------------------------

    logger.info("Plotting final train-validation-test loss comparison")

    plot_final_loss_comparison(
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        final_test_loss=final_test_loss,
        output_dir=OUTPUT_DIR,
    )

    logger.info("Final loss comparison plot saved")

    # -------------------------------------------------------------------------
    # Save run metadata.
    # -------------------------------------------------------------------------

    run_metadata = {
        "model_path": str(MODEL_PATH),
        "analysis_bundle_path": str(
            ANALYSIS_BUNDLE_PATH
        ),
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "use_class_weighted_loss": (
            USE_CLASS_WEIGHTED_LOSS
        ),
        "positive_class_weight": (
            positive_weight_value
        ),
        "final_train_loss": (
            final_train_loss
        ),
        "final_validation_loss": (
            final_validation_loss
        ),
        "final_test_loss": (
            final_test_loss
        ),
        "important_limitation": (
            "Validation and test losses at every epoch "
            "cannot be reconstructed because the original "
            "run did not save those values or every epoch's "
            "model weights. The plotted validation and test "
            "losses are final values for the saved best model."
        ),
    }

    metadata_path = (
        OUTPUT_DIR
        / "baseline_final_loss_run_metadata_v3.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            run_metadata,
            output_file,
            indent=2,
        )

    logger.info("Outputs saved in: %s", OUTPUT_DIR.resolve())
    logger.info("Created files:")

    for output_path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        logger.info(
            "  - %s",
            output_path.name,
        )

    logger.info(
        "Important: validation and test losses shown "
        "are final values for the saved best model, "
        "not per-epoch curves."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error(
            "ERROR: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise
