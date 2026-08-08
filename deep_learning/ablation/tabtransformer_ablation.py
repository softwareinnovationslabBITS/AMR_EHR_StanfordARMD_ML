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
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset


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


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # These settings favor reproducibility over maximum speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 3. MODEL DEFINITION — MATCHES THE ORIGINAL TABTRANSFORMER
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
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
    """TabTransformer supporting zero selected features in any feature family."""

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
        self.n_cat = len(cat_cardinalities)
        self.n_cont = n_cont
        self.n_bin = n_bin

        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(cardinality, cat_embed_dims[name]),
                    nn.Linear(cat_embed_dims[name], attn_embed_dim),
                )
                for name, cardinality in cat_cardinalities.items()
            ]
        )

        self.transformer_layers = (
            nn.Sequential(
                *[
                    TransformerBlock(
                        attn_embed_dim, num_heads, ff_dim, dropout
                    )
                    for _ in range(num_transformer_layers)
                ]
            )
            if self.n_cat > 0
            else None
        )

        if self.n_cont > 0:
            self.cont_bn = nn.BatchNorm1d(self.n_cont)
            self.cont_proj = nn.Sequential(
                nn.Linear(self.n_cont, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 128),
            )
        else:
            self.cont_bn = None
            self.cont_proj = None

        if self.n_bin > 0:
            self.bin_proj = nn.Sequential(
                nn.Linear(self.n_bin, 64),
                nn.GELU(),
            )
        else:
            self.bin_proj = None

        # Wide branch receives all retained continuous and binary features.
        wide_input_dim = self.n_cont + self.n_bin
        self.wide_proj = (
            nn.Linear(wide_input_dim, 64) if wide_input_dim > 0 else None
        )

        n_cat_out = self.n_cat * attn_embed_dim
        n_cont_out = 128 if self.n_cont > 0 else 0
        n_bin_out = 64 if self.n_bin > 0 else 0
        n_wide_out = 64 if wide_input_dim > 0 else 0
        fused_dim = n_cat_out + n_cont_out + n_bin_out + n_wide_out

        if fused_dim == 0:
            raise ValueError("An ablation cannot remove every available feature.")

        mlp_layers: List[nn.Module] = []
        current_dim = fused_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        x_bin: torch.Tensor,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []

        if self.n_cat > 0:
            embedded = [
                embedding(x_cat[:, index])
                for index, embedding in enumerate(self.embeddings)
            ]
            cat_sequence = torch.stack(embedded, dim=1)
            cat_sequence = self.transformer_layers(cat_sequence)
            outputs.append(cat_sequence.flatten(1))

        if self.n_cont > 0:
            normalized_cont = self.cont_bn(x_cont)
            outputs.append(self.cont_proj(normalized_cont))
        else:
            normalized_cont = x_cont

        if self.n_bin > 0:
            outputs.append(self.bin_proj(x_bin))

        wide_parts: List[torch.Tensor] = []
        if self.n_cont > 0:
            wide_parts.append(normalized_cont)
        if self.n_bin > 0:
            wide_parts.append(x_bin)
        if wide_parts:
            outputs.append(self.wide_proj(torch.cat(wide_parts, dim=1)))

        fused = torch.cat(outputs, dim=1)
        return self.mlp(fused).squeeze(1)


# =============================================================================
# 4. DATASET AND DATA HELPERS
# =============================================================================

class AblationDataset(Dataset):
    """Stores only the columns retained for the current ablation experiment."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cat_indices: Sequence[int],
        cont_indices: Sequence[int],
        bin_indices: Sequence[int],
    ):
        n_rows = len(y)

        self.x_cat = (
            torch.from_numpy(
                np.ascontiguousarray(X[:, cat_indices], dtype=np.int64)
            )
            if cat_indices
            else torch.empty((n_rows, 0), dtype=torch.long)
        )
        self.x_cont = (
            torch.from_numpy(
                np.ascontiguousarray(X[:, cont_indices], dtype=np.float32)
            )
            if cont_indices
            else torch.empty((n_rows, 0), dtype=torch.float32)
        )
        self.x_bin = (
            torch.from_numpy(
                np.ascontiguousarray(X[:, bin_indices], dtype=np.float32)
            )
            if bin_indices
            else torch.empty((n_rows, 0), dtype=torch.float32)
        )
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x_cat[index], self.x_cont[index], self.x_bin[index], self.y[index]


def stratified_subsample_indices(
    y: np.ndarray, requested_n: int, seed: int
) -> np.ndarray:
    """Return a reproducible approximately stratified subset without sklearn."""
    if requested_n >= len(y):
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    selected_parts = []
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        label_n = max(1, int(round(requested_n * len(label_indices) / len(y))))
        label_n = min(label_n, len(label_indices))
        selected_parts.append(rng.choice(label_indices, size=label_n, replace=False))

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    return selected[:requested_n]


def maybe_apply_fast_mode(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    if not FAST_MODE:
        return X_train, y_train, X_val, y_val, X_test, y_test

    train_idx = stratified_subsample_indices(y_train, FAST_TRAIN_N, RANDOM_SEED)
    val_idx = stratified_subsample_indices(y_val, FAST_VAL_N, RANDOM_SEED + 1)
    test_idx = stratified_subsample_indices(y_test, FAST_TEST_N, RANDOM_SEED + 2)
    return (
        X_train[train_idx], y_train[train_idx],
        X_val[val_idx], y_val[val_idx],
        X_test[test_idx], y_test[test_idx],
    )


# =============================================================================
# 5. FEATURE GROUP DEFINITIONS
# =============================================================================

def build_feature_groups(
    cat_features: Sequence[str],
    cont_features: Sequence[str],
    binary_features: Sequence[str],
) -> Dict[str, List[str]]:
    """
    Build clinically interpretable groups from the exact saved feature names.

    Prefix-driven groups automatically include all one-hot columns. Features not
    captured by a named group remain in the full model and are documented in the
    feature-group output table.
    """
    all_features = list(cat_features) + list(cont_features) + list(binary_features)

    def existing(names: Sequence[str]) -> List[str]:
        return [name for name in names if name in all_features]

    def starts_with(*prefixes: str) -> List[str]:
        return [name for name in all_features if name.startswith(prefixes)]

    lab_names = [
        name for name in cont_features
        if any(
            token in name
            for token in [
                "wbc", "neutroph", "lymph", "hgb", "plt", "_na",
                "hco3", "bun", "_cr", "lactate", "procalcitonin",
            ]
        )
    ]
    vital_names = [
        name for name in cont_features
        if any(
            token in name
            for token in ["heartrate", "resprate", "temp", "sysbp", "diasbp"]
        )
    ]

    groups = {
        "MICROBIOLOGY_IDENTITY": existing(
            ["organism_enc", "antibiotic_enc", "culture_type_enc"]
        ),
        "CARE_CONTEXT": existing(
            ["ordering_mode_enc", "hosp_ward_IP", "hosp_ward_OP", "hosp_ward_ER", "hosp_ward_ICU"]
        ),
        "DEMOGRAPHICS": existing(["age_enc", "gender_enc"]),
        "TEMPORAL": existing(["order_year", "order_month"]),
        "LABS": lab_names,
        "VITALS": vital_names,
        "COMORBIDITIES": existing(["comorb_total_count"]) + starts_with("comorb_"),
        "ANTIBIOTIC_EXPOSURE": starts_with("abclass_", "absub_"),
        "PRIOR_ORGANISMS": starts_with("priororg_"),
        "PRIOR_PROCEDURES": starts_with("proc_"),
        "SOCIOECONOMIC": existing(["adi_score", "adi_state_rank"]),
        "NURSING_HOME": existing(["nursing_home_visits"]),
        "PRIOR_RESISTANCE": existing(
            ["prior_resistance_count", "min_resistance_days"]
        ),
        "ALL_CATEGORICAL": list(cat_features),
        "ALL_CONTINUOUS": list(cont_features),
        "ALL_BINARY": list(binary_features),
    }

    # Remove duplicates within each group while preserving order.
    groups = {
        group: list(dict.fromkeys(features))
        for group, features in groups.items()
        if len(features) > 0
    }
    return groups


def create_experiments(feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    experiments: Dict[str, List[str]] = {"FULL_CONTROL": []}
    for group_name, feature_names in feature_groups.items():
        experiments[f"NO_{group_name}"] = feature_names
    return experiments


# =============================================================================
# 6. TRAINING, THRESHOLD SELECTION, AND EVALUATION
# =============================================================================

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


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    for x_cat, x_cont, x_bin, y_batch in loader:
        x_cat = x_cat.to(device, non_blocking=True)
        x_cont = x_cont.to(device, non_blocking=True)
        x_bin = x_bin.to(device, non_blocking=True)
        logits = model(x_cat, x_cont, x_bin)
        probability_parts.append(torch.sigmoid(logits).cpu().numpy())
        label_parts.append(y_batch.numpy())

    return np.concatenate(probability_parts), np.concatenate(label_parts)


def select_threshold_on_validation(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, float]:
    """Select the F1-maximizing threshold using validation data only."""
    thresholds = np.linspace(0.01, 0.99, 199)
    f1_values = [
        f1_score(labels, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def calculate_test_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float | int]:
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


def train_one_experiment(
    experiment_name: str,
    removed_features: Sequence[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    all_features: Sequence[str],
    cat_features: Sequence[str],
    cont_features: Sequence[str],
    binary_features: Sequence[str],
    original_cat_cardinalities: Dict[str, int],
    original_cat_embed_dims: Dict[str, int],
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[ExperimentMetrics, List[Dict[str, float]]]:
    set_global_seed(RANDOM_SEED)
    start_time = time.time()

    removed_set = set(removed_features)
    retained_cat = [name for name in cat_features if name not in removed_set]
    retained_cont = [name for name in cont_features if name not in removed_set]
    retained_binary = [name for name in binary_features if name not in removed_set]
    retained_features = retained_cat + retained_cont + retained_binary

    if not retained_features:
        raise ValueError(f"{experiment_name} removes every feature.")

    feature_to_original_index = {
        feature: index for index, feature in enumerate(all_features)
    }
    cat_indices = [feature_to_original_index[name] for name in retained_cat]
    cont_indices = [feature_to_original_index[name] for name in retained_cont]
    bin_indices = [feature_to_original_index[name] for name in retained_binary]

    logger.info(
        "%s | removed=%d | retained=%d (cat=%d, cont=%d, binary=%d)",
        experiment_name,
        len(removed_set),
        len(retained_features),
        len(retained_cat),
        len(retained_cont),
        len(retained_binary),
    )

    train_dataset = AblationDataset(
        X_train, y_train, cat_indices, cont_indices, bin_indices
    )
    val_dataset = AblationDataset(
        X_val, y_val, cat_indices, cont_indices, bin_indices
    )
    test_dataset = AblationDataset(
        X_test, y_test, cat_indices, cont_indices, bin_indices
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    selected_cardinalities = {
        name: original_cat_cardinalities[name] for name in retained_cat
    }
    selected_embedding_dims = {
        name: original_cat_embed_dims[name] for name in retained_cat
    }

    model = AMRTabTransformer(
        cat_cardinalities=selected_cardinalities,
        cat_embed_dims=selected_embedding_dims,
        n_cont=len(retained_cont),
        n_bin=len(retained_binary),
    ).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.astype(int),
    )
    positive_weight = torch.tensor(
        class_weights[1] / class_weights[0],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    use_amp = USE_MIXED_PRECISION and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs_to_run = FAST_EPOCHS if FAST_MODE else NUM_EPOCHS
    best_val_auc = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None
    history_rows: List[Dict[str, float]] = []

    for epoch in range(1, epochs_to_run + 1):
        model.train()
        running_loss = 0.0

        for x_cat, x_cont, x_bin, y_batch in train_loader:
            x_cat = x_cat.to(device, non_blocking=True)
            x_cont = x_cont.to(device, non_blocking=True)
            x_bin = x_bin.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(x_cat, x_cont, x_bin)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(y_batch)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        val_probabilities, val_labels = predict_probabilities(
            model, val_loader, device
        )
        val_auc = roc_auc_score(val_labels, val_probabilities)

        history_rows.append(
            {
                "experiment": experiment_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_roc_auc": val_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        logger.info(
            "%s | epoch %02d/%02d | loss=%.5f | val_auc=%.5f",
            experiment_name,
            epoch,
            epochs_to_run,
            train_loss,
            val_auc,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(
                    "%s | early stopping at epoch %d",
                    experiment_name,
                    epoch,
                )
                break

    if best_state_dict is None:
        raise RuntimeError(f"No valid model state was obtained for {experiment_name}.")

    model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    # The threshold is selected on validation data, never on test data.
    val_probabilities, val_labels = predict_probabilities(model, val_loader, device)
    best_threshold, best_val_f1 = select_threshold_on_validation(
        val_labels, val_probabilities
    )
    logger.info(
        "%s | validation threshold=%.4f | validation F1=%.5f",
        experiment_name,
        best_threshold,
        best_val_f1,
    )

    test_probabilities, test_labels = predict_probabilities(model, test_loader, device)
    test_metrics = calculate_test_metrics(
        test_labels, test_probabilities, best_threshold
    )

    elapsed_minutes = (time.time() - start_time) / 60.0
    removed_group = (
        "NONE" if experiment_name == "FULL_CONTROL"
        else experiment_name.removeprefix("NO_")
    )

    metrics = ExperimentMetrics(
        experiment=experiment_name,
        removed_group=removed_group,
        n_removed_features=len(removed_set),
        n_retained_features=len(retained_features),
        n_cat_features=len(retained_cat),
        n_cont_features=len(retained_cont),
        n_binary_features=len(retained_binary),
        best_epoch=best_epoch,
        best_val_roc_auc=float(best_val_auc),
        validation_threshold=best_threshold,
        training_minutes=elapsed_minutes,
        **test_metrics,
    )

    if SAVE_EACH_MODEL:
        model_path = MODEL_OUTPUT_DIR / f"ablation_weights_{experiment_name.lower()}.pt"
        torch.save(
            {
                "experiment": experiment_name,
                "removed_features": list(removed_features),
                "retained_cat_features": retained_cat,
                "retained_cont_features": retained_cont,
                "retained_binary_features": retained_binary,
                "model_state_dict": best_state_dict,
                "cat_cardinalities": selected_cardinalities,
                "cat_embed_dims": selected_embedding_dims,
                "n_cont": len(retained_cont),
                "n_bin": len(retained_binary),
                "validation_threshold": best_threshold,
                "metrics": asdict(metrics),
            },
            model_path,
        )

    # Release model- and experiment-specific memory before the next ablation.
    del model, optimizer, scheduler, criterion, scaler
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    del best_state_dict, val_probabilities, val_labels
    del test_probabilities, test_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, history_rows


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
