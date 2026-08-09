#!/usr/bin/env python3
# Source: /AMR_Stanford/DL_codes/amr_project/amr_tabtransformer_ablation_study.py
"""
AMR TabTransformer — Feature-group ablation study
==================================================

This script performs a *retraining-based* ablation study. For every experiment,
a new TabTransformer is trained from scratch after removing one clinically
meaningful feature group. This is more defensible than simply replacing values
in the already-trained full model, because the reduced model is allowed to
relearn its parameters without the removed information.

Required input artifacts (created by the existing training script):
    amr_model.pt
    amr_analysis_bundle.joblib

All ablation outputs are written to a new directory:
    ./amr_ablation_study_outputs/

Main outputs:
    ablation_summary_metrics.csv
    ablation_delta_from_control.csv
    ablation_training_history.csv
    ablation_feature_groups.csv
    ablation_run_config.json
    ablation_performance_comparison.png
    ablation_auc_drop.png
    ablation_training_curves.png
    logs/ablation_study.log
    models/*.pt                     (optional)

Important methodology:
1. The exact saved train/validation/test arrays are reused.
2. The same random seed and training settings are used for all experiments.
3. The classification threshold is selected on the validation set only.
4. Final metrics are calculated once on the untouched test set.
5. "FULL_CONTROL" is retrained from scratch for a fair comparison.
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# #migrate: split model, feature groups, metrics, and training into private modules
from ablation_feature_groups import build_feature_groups, create_experiments
from ablation_metrics import ExperimentMetrics
from ablation_training import maybe_apply_fast_mode, set_global_seed, train_one_experiment


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# #migrate: load settings from the single config file
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
_TT_CFG = _CFG.get('tabtransformer', {})

# #migrate: artifact paths updated for repo structure
MODEL_BUNDLE_PATH = resolve_path(_TT_CFG.get('model_path', 'models/tabtransformer/amr_model.pt'))
ANALYSIS_BUNDLE_PATH = resolve_path(_TT_CFG.get('bundle_path', 'dataset/amr_analysis_bundle.joblib'))

# #migrate: output directory from the single config file
OUTPUT_DIR = Path(str(resolve_path(_TT_CFG.get('ablation_output_dir', 'deep_learning/amr_ablation_study_outputs'))))
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"

# Reproducibility.
RANDOM_SEED = _CFG.get('seed', 42)

# Training settings. These mirror the original pipeline where possible.
BATCH_SIZE = _TT_CFG.get('batch_size', 512)
NUM_EPOCHS = _TT_CFG.get('epochs', 60)
EARLY_STOPPING_PATIENCE = _TT_CFG.get('patience', 10)
LEARNING_RATE = _TT_CFG.get('learning_rate', 3e-4)
WEIGHT_DECAY = _TT_CFG.get('weight_decay', 1e-4)
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 0

# Mixed precision reduces GPU memory use and generally speeds CUDA training.
# It is automatically disabled on CPU.
USE_MIXED_PRECISION = True

# Set True to save the best weights for every ablation model. This may use
# substantial disk space. Metrics and histories are always saved.
SAVE_EACH_MODEL = False

# Resume mode skips an experiment already present in the summary CSV.
RESUME_IF_POSSIBLE = True

# Optional development mode. When True, only a stratified sample of each split
# is used and epochs are reduced. Never use FAST_MODE results in the manuscript.
FAST_MODE = False
FAST_TRAIN_N = 100_000
FAST_VAL_N = 30_000
FAST_TEST_N = 30_000
FAST_EPOCHS = 3

# Run all defined ablations. To run a subset, place names here, for example:
# ONLY_EXPERIMENTS = ["FULL_CONTROL", "NO_LABS", "NO_PRIOR_RESISTANCE"]
ONLY_EXPERIMENTS: List[str] | None = None


# =============================================================================
# 2. LOGGING AND REPRODUCIBILITY
# =============================================================================

def create_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("amr_ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_DIR / "ablation_study.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


# =============================================================================
# 3-6. MODEL, DATA HELPERS, FEATURE GROUPS, AND TRAINING LOOP
# =============================================================================
# #migrate: these sections now live in ablation_model.py, ablation_feature_groups.py,
# ablation_metrics.py, and ablation_training.py.


# =============================================================================
# 4. DATASET AND DATA HELPERS
# =============================================================================
# #migrate: imported from ablation_training.py


# =============================================================================
# 5. FEATURE GROUP DEFINITIONS
# =============================================================================
# #migrate: imported from ablation_feature_groups.py


# =============================================================================
# 6. TRAINING, THRESHOLD SELECTION, AND EVALUATION
# =============================================================================
# #migrate: imported from ablation_metrics.py and ablation_training.py


# =============================================================================
# 7. OUTPUT TABLES AND FIGURES
# =============================================================================

def save_feature_group_manifest(
    feature_groups: Dict[str, List[str]], all_features: Sequence[str]
) -> None:
    rows = []
    grouped_features = set()
    for group_name, feature_names in feature_groups.items():
        for feature_name in feature_names:
            rows.append(
                {
                    "feature_group": group_name,
                    "feature_name": feature_name,
                }
            )
            grouped_features.add(feature_name)

    for feature_name in all_features:
        if feature_name not in grouped_features:
            rows.append(
                {
                    "feature_group": "NOT_IN_NAMED_CLINICAL_GROUP",
                    "feature_name": feature_name,
                }
            )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "ablation_feature_groups.csv", index=False
    )


def save_current_results(
    metrics_rows: List[Dict], history_rows: List[Dict]
) -> None:
    pd.DataFrame(metrics_rows).to_csv(
        OUTPUT_DIR / "ablation_summary_metrics.csv", index=False
    )
    pd.DataFrame(history_rows).to_csv(
        OUTPUT_DIR / "ablation_training_history.csv", index=False
    )


def calculate_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    control_rows = summary.loc[summary["experiment"] == "FULL_CONTROL"]
    if control_rows.empty:
        raise ValueError("FULL_CONTROL is required to calculate ablation deltas.")
    control = control_rows.iloc[0]

    delta_columns = [
        "test_roc_auc",
        "test_pr_auc",
        "test_balanced_accuracy",
        "test_mcc",
        "test_cohen_kappa",
        "test_f1",
        "test_brier_score",
    ]

    delta = summary.copy()
    for column in delta_columns:
        # Positive drop means the ablation performed worse than the control.
        if column == "test_brier_score":
            delta[f"increase_{column}_vs_control"] = delta[column] - control[column]
        else:
            delta[f"drop_{column}_vs_control"] = control[column] - delta[column]

    return delta


def create_performance_figures(summary: pd.DataFrame, history: pd.DataFrame) -> None:
    plot_data = summary.sort_values("test_roc_auc", ascending=True).copy()

    # Figure 1: ROC-AUC and PR-AUC across experiments.
    y_positions = np.arange(len(plot_data))
    fig, ax = plt.subplots(figsize=(12, max(6, 0.48 * len(plot_data))))
    ax.plot(plot_data["test_roc_auc"], y_positions, "o-", label="ROC-AUC")
    ax.plot(plot_data["test_pr_auc"], y_positions, "s-", label="PR-AUC")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["experiment"])
    ax.set_xlabel("Test-set score")
    ax.set_title("TabTransformer Ablation Performance")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ablation_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: ROC-AUC drop relative to the retrained full control.
    control_auc = float(
        summary.loc[summary["experiment"] == "FULL_CONTROL", "test_roc_auc"].iloc[0]
    )
    drop_data = summary.loc[summary["experiment"] != "FULL_CONTROL"].copy()
    drop_data["roc_auc_drop"] = control_auc - drop_data["test_roc_auc"]
    drop_data = drop_data.sort_values("roc_auc_drop", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.48 * len(drop_data))))
    ax.barh(drop_data["experiment"], drop_data["roc_auc_drop"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("ROC-AUC drop versus FULL_CONTROL")
    ax.set_title("Effect of Removing Each Feature Group")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ablation_auc_drop.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: validation AUC curves for every experiment.
    if not history.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        for experiment, subset in history.groupby("experiment", sort=False):
            ax.plot(
                subset["epoch"],
                subset["validation_roc_auc"],
                label=experiment,
                linewidth=1.5,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation ROC-AUC")
        ax.set_title("Ablation Training Curves")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "ablation_training_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


# =============================================================================
# 8. MAIN PROGRAM
# =============================================================================

def main() -> None:
    create_output_directories()
    logger = configure_logging()
    set_global_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting AMR TabTransformer ablation study")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("CUDA is unavailable; training will run on CPU and may be slow.")

    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {MODEL_BUNDLE_PATH.resolve()}")
    if not ANALYSIS_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {ANALYSIS_BUNDLE_PATH.resolve()}")

    logger.info("Loading %s", MODEL_BUNDLE_PATH)
    model_bundle = torch.load(
        MODEL_BUNDLE_PATH, map_location="cpu", weights_only=False
    )
    logger.info("Loading %s", ANALYSIS_BUNDLE_PATH)
    analysis_bundle = joblib.load(ANALYSIS_BUNDLE_PATH)

    required_model_keys = {
        "cat_cardinalities",
        "cat_embed_dims",
    }
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
    missing_model_keys = required_model_keys - set(model_bundle)
    missing_analysis_keys = required_analysis_keys - set(analysis_bundle)
    if missing_model_keys:
        raise KeyError(f"Model bundle is missing keys: {sorted(missing_model_keys)}")
    if missing_analysis_keys:
        raise KeyError(
            f"Analysis bundle is missing keys: {sorted(missing_analysis_keys)}"
        )

    cat_features = list(analysis_bundle["CAT_FEATURES"])
    cont_features = list(analysis_bundle["CONT_FEATURES"])
    binary_features = list(analysis_bundle["BINARY_FEATURES"])
    all_features = list(analysis_bundle["ALL_FEATURES"])

    expected_order = cat_features + cont_features + binary_features
    if all_features != expected_order:
        raise ValueError(
            "ALL_FEATURES is not ordered as CAT_FEATURES + CONT_FEATURES + "
            "BINARY_FEATURES. The script stops to avoid selecting wrong columns."
        )

    X_train = np.asarray(analysis_bundle["X_train"], dtype=np.float32)
    X_val = np.asarray(analysis_bundle["X_val"], dtype=np.float32)
    X_test = np.asarray(analysis_bundle["X_test"], dtype=np.float32)
    y_train = np.asarray(analysis_bundle["y_train"], dtype=np.float32)
    y_val = np.asarray(analysis_bundle["y_val"], dtype=np.float32)
    y_test = np.asarray(analysis_bundle["y_test"], dtype=np.float32)

    X_train, y_train, X_val, y_val, X_test, y_test = maybe_apply_fast_mode(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    logger.info(
        "Data sizes | train=%s | validation=%s | test=%s | features=%d",
        f"{len(y_train):,}",
        f"{len(y_val):,}",
        f"{len(y_test):,}",
        len(all_features),
    )

    feature_groups = build_feature_groups(
        cat_features, cont_features, binary_features
    )
    experiments = create_experiments(feature_groups)

    if ONLY_EXPERIMENTS is not None:
        unknown = set(ONLY_EXPERIMENTS) - set(experiments)
        if unknown:
            raise ValueError(f"Unknown experiments requested: {sorted(unknown)}")
        experiments = {
            name: experiments[name] for name in ONLY_EXPERIMENTS
        }
        if "FULL_CONTROL" not in experiments:
            logger.warning(
                "FULL_CONTROL was not selected; delta calculations will be unavailable."
            )

    save_feature_group_manifest(feature_groups, all_features)

    run_config = {
        "model_bundle_path": str(MODEL_BUNDLE_PATH),
        "analysis_bundle_path": str(ANALYSIS_BUNDLE_PATH),
        "output_directory": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs": FAST_EPOCHS if FAST_MODE else NUM_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "mixed_precision": USE_MIXED_PRECISION and device.type == "cuda",
        "fast_mode": FAST_MODE,
        "save_each_model": SAVE_EACH_MODEL,
        "device": str(device),
        "experiments": {
            name: removed for name, removed in experiments.items()
        },
    }
    with open(OUTPUT_DIR / "ablation_run_config.json", "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    metrics_rows: List[Dict] = []
    history_rows: List[Dict] = []
    completed_experiments = set()

    summary_path = OUTPUT_DIR / "ablation_summary_metrics.csv"
    history_path = OUTPUT_DIR / "ablation_training_history.csv"
    if RESUME_IF_POSSIBLE and summary_path.exists():
        old_summary = pd.read_csv(summary_path)
        metrics_rows = old_summary.to_dict("records")
        completed_experiments = set(old_summary["experiment"].astype(str))
        logger.info(
            "Resume enabled: found %d completed experiments.",
            len(completed_experiments),
        )
        if history_path.exists():
            history_rows = pd.read_csv(history_path).to_dict("records")

    for experiment_name, removed_features in experiments.items():
        if experiment_name in completed_experiments:
            logger.info("Skipping completed experiment: %s", experiment_name)
            continue

        logger.info("=" * 80)
        logger.info("Running experiment: %s", experiment_name)
        try:
            # #migrate: pass the model output directory to the training helper
            metrics, experiment_history = train_one_experiment(
                experiment_name=experiment_name,
                removed_features=removed_features,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                all_features=all_features,
                cat_features=cat_features,
                cont_features=cont_features,
                binary_features=binary_features,
                original_cat_cardinalities=model_bundle["cat_cardinalities"],
                original_cat_embed_dims=model_bundle["cat_embed_dims"],
                device=device,
                logger=logger,
                model_output_dir=MODEL_OUTPUT_DIR,
            )
            metrics_rows.append(asdict(metrics))
            history_rows.extend(experiment_history)
            save_current_results(metrics_rows, history_rows)
            logger.info(
                "%s complete | test ROC-AUC=%.5f | PR-AUC=%.5f | MCC=%.5f",
                experiment_name,
                metrics.test_roc_auc,
                metrics.test_pr_auc,
                metrics.test_mcc,
            )
        except Exception:
            logger.exception("Experiment failed: %s", experiment_name)
            # Preserve prior successful results, then stop. This avoids silently
            # producing an incomplete manuscript table without notice.
            save_current_results(metrics_rows, history_rows)
            raise

    summary = pd.DataFrame(metrics_rows)
    history = pd.DataFrame(history_rows)

    if summary.empty:
        raise RuntimeError("No experiments were completed.")

    # Preserve requested experiment order in the final tables.
    experiment_order = list(experiments)
    summary["_order"] = summary["experiment"].map(
        {name: index for index, name in enumerate(experiment_order)}
    )
    summary = summary.sort_values("_order").drop(columns="_order")
    summary.to_csv(summary_path, index=False)

    if "FULL_CONTROL" in set(summary["experiment"]):
        delta = calculate_deltas(summary)
        delta.to_csv(
            OUTPUT_DIR / "ablation_delta_from_control.csv", index=False
        )
        create_performance_figures(summary, history)
    else:
        logger.warning(
            "FULL_CONTROL is absent, so control deltas and comparison figures were skipped."
        )

    logger.info("=" * 80)
    logger.info("Ablation study finished successfully.")
    logger.info("Results directory: %s", OUTPUT_DIR.resolve())
    logger.info("Main metrics: %s", summary_path.resolve())


if __name__ == "__main__":
    main()
