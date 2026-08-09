#!/usr/bin/env python3
# Source: /AMR_Stanford/DL_codes/amr_project/baseline_bootstrap_ci.py
"""
Standalone bootstrap confidence-interval analysis for the baseline AMR
TabTransformer model.

This script DOES NOT retrain the model. It:
  1. Loads the saved baseline model from amr_model.pt.
  2. Loads the exact validation and test arrays from amr_analysis_bundle.joblib.
  3. Generates validation and test probabilities.
  4. Selects one classification threshold using the validation set only.
  5. Calculates point estimates on the untouched test set.
  6. Uses stratified bootstrap resampling of paired test labels/probabilities.
  7. Saves confidence intervals, all bootstrap replicates, predictions, and plots
     under a new output directory with names distinct from previous analyses.

Required input files (unchanged names):
  - amr_model.pt
  - amr_analysis_bundle.joblib

Main outputs:
  - baseline_ci_bootstrap_results_v2/baseline_ci_summary_v2.csv
  - baseline_ci_bootstrap_results_v2/baseline_ci_manuscript_table_v2.csv
  - baseline_ci_bootstrap_results_v2/baseline_bootstrap_replicates_v2.csv.gz
  - baseline_ci_bootstrap_results_v2/baseline_test_predictions_v2.npz
  - baseline_ci_bootstrap_results_v2/baseline_ci_forest_plot_v2.png/.pdf
  - baseline_ci_bootstrap_results_v2/baseline_bootstrap_distributions_v2.png/.pdf

Methodological note:
The threshold is optimized once on the validation set and then held fixed for
all test-set analyses and bootstrap replicates. This prevents threshold tuning
on the test data.
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
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

# #migrate: split bootstrap CI and plotting helpers into private modules
from ablation_bootstrap import run_stratified_bootstrap, summarize_bootstrap
from ablation_plot_results import create_manuscript_table, make_distribution_plot, make_forest_plot

# #migrate: configure timestamped progress logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# USER SETTINGS
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
OUTPUT_DIR = Path(str(resolve_path(_TT_CFG.get('bootstrap_ci_output_dir', 'deep_learning/baseline_ci_bootstrap_results_v2'))))

# Recommended final analysis: 2,000. Use 100 during an initial test run.
N_BOOTSTRAPS = 2000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260806

# Inference settings. Increase batch size if GPU memory permits.
PREDICTION_BATCH_SIZE = 1024
NUM_WORKERS = 0
PIN_MEMORY = True

# Threshold-selection rule. Currently supports "f1" or "youden".
THRESHOLD_METHOD = "f1"

# Print progress every N bootstrap iterations.
PROGRESS_EVERY = 50

# Save all replicate-level metrics as a compressed CSV.
SAVE_ALL_BOOTSTRAP_REPLICATES = True

# If True, a small diagnostic run is performed. Do not report FAST_MODE output.
FAST_MODE = False
FAST_BOOTSTRAPS = 100
FAST_MAX_VAL_ROWS = 50_000
FAST_MAX_TEST_ROWS = 50_000


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_reproducibility(seed: int) -> None:
    """Set random seeds used by NumPy, Python, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# MODEL DEFINITION — MUST MATCH THE ORIGINAL TRAINING SCRIPT
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
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
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
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
        return self.norm(x + self.dropout(self.ff(x)))


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
    ):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(cardinality, cat_embed_dims[feature]),
                    nn.Linear(cat_embed_dims[feature], attn_embed_dim),
                )
                for feature, cardinality in cat_cardinalities.items()
            ]
        )

        self.transformer_layers = nn.Sequential(
            *[
                TransformerBlock(
                    attn_embed_dim,
                    num_heads,
                    ff_dim,
                    dropout,
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

        self.wide_proj = nn.Linear(n_cont + n_bin, 64)

        n_cat = len(cat_cardinalities) * attn_embed_dim
        n_deep = 128 + (64 if n_bin > 0 else 0)
        total_input_dim = n_cat + n_deep + 64

        mlp_layers = []
        in_dim = total_input_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        x_bin: torch.Tensor,
    ) -> torch.Tensor:
        cat_embeddings = [
            embedding(x_cat[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        cat_sequence = torch.stack(cat_embeddings, dim=1)
        cat_sequence = self.transformer_layers(cat_sequence)
        cat_flat = cat_sequence.flatten(1)

        x_cont_normalized = self.cont_bn(x_cont)
        continuous_output = self.cont_proj(x_cont_normalized)

        if self.n_bin > 0:
            binary_output = self.bin_proj(x_bin)
            deep_output = torch.cat(
                [continuous_output, binary_output],
                dim=1,
            )
        else:
            deep_output = continuous_output

        wide_input = torch.cat([x_cont_normalized, x_bin], dim=1)
        wide_output = self.wide_proj(wide_input)

        fused = torch.cat([cat_flat, deep_output, wide_output], dim=1)
        return self.mlp(fused).squeeze(1)


# =============================================================================
# DATASET AND INFERENCE
# =============================================================================

class AMRInferenceDataset(Dataset):
    """Dataset holding only the arrays required for model inference."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: Iterable[int],
        cont_idx: Iterable[int],
        bin_idx: Iterable[int],
    ):
        cat_idx = list(cat_idx)
        cont_idx = list(cont_idx)
        bin_idx = list(bin_idx)

        self.X_cat = torch.as_tensor(
            X[:, cat_idx].astype(np.int64, copy=False),
            dtype=torch.long,
        )
        self.X_cont = torch.as_tensor(
            X[:, cont_idx].astype(np.float32, copy=False),
            dtype=torch.float32,
        )
        self.X_bin = torch.as_tensor(
            X[:, bin_idx].astype(np.float32, copy=False),
            dtype=torch.float32,
        )
        self.y = torch.as_tensor(
            y.astype(np.float32, copy=False),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            self.X_cat[index],
            self.X_cont[index],
            self.X_bin[index],
            self.y[index],
        )


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate class-1 probabilities and return aligned labels."""
    model.eval()
    probability_batches = []
    label_batches = []

    for x_cat, x_cont, x_bin, y_batch in loader:
        x_cat = x_cat.to(device, non_blocking=True)
        x_cont = x_cont.to(device, non_blocking=True)
        x_bin = x_bin.to(device, non_blocking=True)

        logits = model(x_cat, x_cont, x_bin)
        probabilities = torch.sigmoid(logits)

        probability_batches.append(probabilities.cpu().numpy())
        label_batches.append(y_batch.numpy())

    return (
        np.concatenate(probability_batches).astype(np.float64),
        np.concatenate(label_batches).astype(np.int8),
    )


# =============================================================================
# THRESHOLD SELECTION AND METRICS
# =============================================================================

def select_validation_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "f1",
) -> Tuple[float, float]:
    """Select a threshold using validation data only."""
    method = method.lower()

    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        if thresholds.size == 0:
            raise RuntimeError("No validation thresholds could be calculated.")

        f1_values = (
            2.0 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_index = int(np.nanargmax(f1_values))
        return float(thresholds[best_index]), float(f1_values[best_index])

    if method == "youden":
        # Avoid importing roc_curve unless this option is selected.
        from sklearn.metrics import roc_curve

        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            y_true,
            y_prob,
        )
        youden_j = true_positive_rate - false_positive_rate
        finite_mask = np.isfinite(thresholds)
        if not finite_mask.any():
            raise RuntimeError("No finite Youden thresholds were available.")
        candidate_indices = np.flatnonzero(finite_mask)
        best_local = int(np.nanargmax(youden_j[finite_mask]))
        best_index = int(candidate_indices[best_local])
        return float(thresholds[best_index]), float(youden_j[best_index])

    raise ValueError("THRESHOLD_METHOD must be either 'f1' or 'youden'.")


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculate all baseline performance metrics."""
    y_true = np.asarray(y_true, dtype=np.int8).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_pred = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# =============================================================================
# BOOTSTRAP
# =============================================================================
# #migrate: imported from ablation_bootstrap.py


# =============================================================================
# OUTPUT TABLES AND PLOTS
# =============================================================================
# #migrate: imported from ablation_plot_results.py


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    set_reproducibility(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Required model file not found: {MODEL_PATH}")
    if not ANALYSIS_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Required analysis bundle not found: {ANALYSIS_BUNDLE_PATH}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Python executable: %s", sys.executable)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Loading saved model and exact validation/test splits")
    model_bundle = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )
    analysis_bundle = joblib.load(ANALYSIS_BUNDLE_PATH)

    required_model_keys = {
        "model_state_dict",
        "cat_cardinalities",
        "cat_embed_dims",
        "n_cont",
        "n_bin",
    }
    required_analysis_keys = {
        "CAT_FEATURES",
        "CONT_FEATURES",
        "BINARY_FEATURES",
        "ALL_FEATURES",
        "X_val",
        "X_test",
        "y_val",
        "y_test",
    }

    missing_model = required_model_keys - set(model_bundle)
    missing_analysis = required_analysis_keys - set(analysis_bundle)
    if missing_model:
        raise KeyError(f"Model bundle is missing keys: {sorted(missing_model)}")
    if missing_analysis:
        raise KeyError(
            f"Analysis bundle is missing keys: {sorted(missing_analysis)}"
        )

    cat_features = analysis_bundle["CAT_FEATURES"]
    cont_features = analysis_bundle["CONT_FEATURES"]
    binary_features = analysis_bundle["BINARY_FEATURES"]
    all_features = analysis_bundle["ALL_FEATURES"]

    # Prefer saved indices when available; otherwise reconstruct them exactly.
    cat_idx = analysis_bundle.get(
        "cat_idx",
        list(range(len(cat_features))),
    )
    cont_idx = analysis_bundle.get(
        "cont_idx",
        list(range(len(cat_features), len(cat_features) + len(cont_features))),
    )
    bin_idx = analysis_bundle.get(
        "bin_idx",
        list(
            range(
                len(cat_features) + len(cont_features),
                len(all_features),
            )
        ),
    )

    X_val = np.asarray(analysis_bundle["X_val"], dtype=np.float32)
    X_test = np.asarray(analysis_bundle["X_test"], dtype=np.float32)
    y_val = np.asarray(analysis_bundle["y_val"], dtype=np.int8).ravel()
    y_test = np.asarray(analysis_bundle["y_test"], dtype=np.int8).ravel()

    n_bootstraps = N_BOOTSTRAPS
    if FAST_MODE:
        logger.warning("FAST_MODE is enabled. Results are diagnostic only.")
        rng_fast = np.random.default_rng(RANDOM_SEED)

        def stratified_subsample(X, y, max_rows):
            if len(y) <= max_rows:
                return X, y
            neg = np.flatnonzero(y == 0)
            pos = np.flatnonzero(y == 1)
            target_pos = max(1, int(round(max_rows * len(pos) / len(y))))
            target_neg = max_rows - target_pos
            chosen = np.concatenate(
                [
                    rng_fast.choice(neg, size=min(target_neg, len(neg)), replace=False),
                    rng_fast.choice(pos, size=min(target_pos, len(pos)), replace=False),
                ]
            )
            rng_fast.shuffle(chosen)
            return X[chosen], y[chosen]

        X_val, y_val = stratified_subsample(X_val, y_val, FAST_MAX_VAL_ROWS)
        X_test, y_test = stratified_subsample(X_test, y_test, FAST_MAX_TEST_ROWS)
        n_bootstraps = FAST_BOOTSTRAPS

    logger.info(
        "Validation: %d rows; positive rate=%.4f",
        len(y_val),
        y_val.mean(),
    )
    logger.info(
        "Test: %d rows; positive rate=%.4f",
        len(y_test),
        y_test.mean(),
    )
    logger.info(
        "Features: categorical=%d, continuous=%d, binary=%d, total=%d",
        len(cat_features),
        len(cont_features),
        len(binary_features),
        len(all_features),
    )

    model = AMRTabTransformer(
        cat_cardinalities=model_bundle["cat_cardinalities"],
        cat_embed_dims=model_bundle["cat_embed_dims"],
        n_cont=int(model_bundle["n_cont"]),
        n_bin=int(model_bundle["n_bin"]),
    ).to(device)
    model.load_state_dict(model_bundle["model_state_dict"])
    model.eval()

    val_dataset = AMRInferenceDataset(
        X_val,
        y_val,
        cat_idx,
        cont_idx,
        bin_idx,
    )
    test_dataset = AMRInferenceDataset(
        X_test,
        y_test,
        cat_idx,
        cont_idx,
        bin_idx,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=PREDICTION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=PREDICTION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and device.type == "cuda",
    )

    logger.info("Generating validation probabilities")
    val_probs, val_labels = predict_probabilities(model, val_loader, device)

    logger.info("Generating test probabilities")
    test_probs, test_labels = predict_probabilities(model, test_loader, device)

    selected_threshold, validation_objective = select_validation_threshold(
        val_labels,
        val_probs,
        THRESHOLD_METHOD,
    )
    logger.info(
        "Validation-selected threshold (%s): %.6f",
        THRESHOLD_METHOD,
        selected_threshold,
    )
    logger.info("Validation objective value: %.6f", validation_objective)

    point_estimates = calculate_metrics(
        test_labels,
        test_probs,
        selected_threshold,
    )

    logger.info("Baseline test-set point estimates calculated")
    for metric, value in point_estimates.items():
        if isinstance(value, float):
            logger.info("  %30s: %.6f", metric, value)
        else:
            logger.info("  %30s: %s", metric, value)

    # Save probabilities before bootstrapping so they can be reused without
    # another model inference pass.
    np.savez_compressed(
        OUTPUT_DIR / "baseline_test_predictions_v2.npz",
        test_labels=test_labels,
        test_probabilities=test_probs,
        validation_labels=val_labels,
        validation_probabilities=val_probs,
        selected_threshold=np.array([selected_threshold]),
    )

    threshold_table = pd.DataFrame(
        [
            {
                "threshold_selection_dataset": "validation",
                "threshold_method": THRESHOLD_METHOD,
                "selected_threshold": selected_threshold,
                "validation_objective_value": validation_objective,
                "validation_n": len(val_labels),
                "test_n": len(test_labels),
            }
        ]
    )
    threshold_table.to_csv(
        OUTPUT_DIR / "baseline_threshold_details_v2.csv",
        index=False,
    )

    logger.info(
        "Starting %d stratified bootstrap replicates",
        n_bootstraps,
    )
    # #migrate: pass metric callback and log progress every 250 iterations
    bootstrap_df = run_stratified_bootstrap(
        y_true=test_labels,
        y_prob=test_probs,
        threshold=selected_threshold,
        calculate_metrics=calculate_metrics,
        n_bootstraps=n_bootstraps,
        seed=RANDOM_SEED,
        progress_every=250,
    )

    summary_df = summarize_bootstrap(
        point_estimates=point_estimates,
        bootstrap_df=bootstrap_df,
        confidence_level=CONFIDENCE_LEVEL,
        threshold=selected_threshold,
        threshold_method=THRESHOLD_METHOD,
    )
    manuscript_df = create_manuscript_table(summary_df)

    summary_df.to_csv(
        OUTPUT_DIR / "baseline_ci_summary_v2.csv",
        index=False,
    )
    manuscript_df.to_csv(
        OUTPUT_DIR / "baseline_ci_manuscript_table_v2.csv",
        index=False,
    )

    confusion_table = pd.DataFrame(
        [
            {
                "selected_threshold": selected_threshold,
                "true_negative": point_estimates["true_negative"],
                "false_positive": point_estimates["false_positive"],
                "false_negative": point_estimates["false_negative"],
                "true_positive": point_estimates["true_positive"],
            }
        ]
    )
    confusion_table.to_csv(
        OUTPUT_DIR / "baseline_confusion_counts_v2.csv",
        index=False,
    )

    if SAVE_ALL_BOOTSTRAP_REPLICATES:
        bootstrap_df.to_csv(
            OUTPUT_DIR / "baseline_bootstrap_replicates_v2.csv.gz",
            index=False,
            compression="gzip",
        )

    run_configuration = {
        "model_path": str(MODEL_PATH),
        "analysis_bundle_path": str(ANALYSIS_BUNDLE_PATH),
        "output_directory": str(OUTPUT_DIR),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "n_bootstraps": n_bootstraps,
        "confidence_level": CONFIDENCE_LEVEL,
        "random_seed": RANDOM_SEED,
        "bootstrap_method": "stratified_percentile",
        "threshold_method": THRESHOLD_METHOD,
        "selected_threshold": selected_threshold,
        "threshold_selected_on": "validation_set",
        "validation_n": int(len(val_labels)),
        "test_n": int(len(test_labels)),
        "test_positive_rate": float(test_labels.mean()),
        "fast_mode": FAST_MODE,
    }
    with open(
        OUTPUT_DIR / "baseline_ci_run_configuration_v2.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_configuration, handle, indent=2)

    logger.info("Creating forest plot")
    make_forest_plot(summary_df, OUTPUT_DIR)
    logger.info("Forest plot saved")

    logger.info("Creating bootstrap distribution plot")
    make_distribution_plot(bootstrap_df, point_estimates, OUTPUT_DIR)
    logger.info("Bootstrap distribution plot saved")

    logger.info("Final manuscript-ready values")
    logger.info(
        "\n%s",
        manuscript_df[["metric_display", "estimate_95_ci"]].to_string(index=False),
    )

    logger.info("All outputs saved under: %s", OUTPUT_DIR.resolve())
    for path in sorted(OUTPUT_DIR.iterdir()):
        logger.info("  - %s", path.name)

    # Explicit cleanup can be helpful on shared GPU servers.
    del model, val_dataset, test_dataset, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
