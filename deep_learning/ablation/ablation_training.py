"""
Training/validation loop helpers for the TabTransformer ablation study.

#migrate: extracted from tabtransformer_ablation.py
"""

import gc
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

# #migrate: load training hyperparameters from the single config file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_loader import load_config

from ablation_metrics import ExperimentMetrics, calculate_test_metrics
from ablation_model import AMRTabTransformer

_CFG = load_config()
_TT_CFG = _CFG.get("tabtransformer", {})

BATCH_SIZE = _TT_CFG.get("batch_size", 512)
NUM_EPOCHS = _TT_CFG.get("epochs", 60)
EARLY_STOPPING_PATIENCE = _TT_CFG.get("patience", 10)
LEARNING_RATE = _TT_CFG.get("learning_rate", 3e-4)
WEIGHT_DECAY = _TT_CFG.get("weight_decay", 1e-4)
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 0
USE_MIXED_PRECISION = True
SAVE_EACH_MODEL = False
RANDOM_SEED = _CFG.get("seed", 42)

FAST_MODE = False
FAST_TRAIN_N = 100_000
FAST_VAL_N = 30_000
FAST_TEST_N = 30_000
FAST_EPOCHS = 3


def set_global_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    """Optionally downsample all splits for a fast development run."""
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


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate class-1 probabilities and return aligned labels."""
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
    """Select the MCC-maximizing threshold using validation data only."""
    thresholds = np.linspace(0.001, 0.999, 999)
    best_mcc = -2.0
    best_threshold = 0.5
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(np.int8)
        mcc = matthews_corrcoef(labels, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    return float(best_threshold), float(best_mcc)


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
    model_output_dir: Path,
) -> Tuple[ExperimentMetrics, List[Dict[str, float]]]:
    """Train one ablation experiment from scratch and return its metrics."""
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

    # #migrate: log feature group being ablated
    logger.info(
        "Running ablation for feature group: %s | removed=%d | retained=%d (cat=%d, cont=%d, binary=%d)",
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
    best_val_loss = float('inf')
    best_val_auc  = -np.inf
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

        # Compute validation loss
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_cont, x_bin, y_batch in val_loader:
                x_cat = x_cat.to(device, non_blocking=True)
                x_cont = x_cont.to(device, non_blocking=True)
                x_bin = x_bin.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x_cat, x_cont, x_bin)
                    val_loss = criterion(logits, y_batch)
                running_val_loss += val_loss.item() * len(y_batch)
        val_loss_epoch = running_val_loss / len(val_dataset)

        val_probabilities, val_labels = predict_probabilities(
            model, val_loader, device
        )
        val_auc = roc_auc_score(val_labels, val_probabilities)
        val_pr_auc = average_precision_score(val_labels, val_probabilities)

        history_rows.append(
            {
                "experiment": experiment_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss_epoch,
                "validation_roc_auc": val_auc,
                "val_pr_auc": val_pr_auc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # #migrate: log epoch progress for each ablation experiment
        logger.info(
            "%s | epoch %02d/%02d | train_loss=%.5f | val_loss=%.5f | val_auc=%.5f | val_pr_auc=%.5f",
            experiment_name,
            epoch,
            epochs_to_run,
            train_loss,
            val_loss_epoch,
            val_auc,
            val_pr_auc,
        )

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
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
                    "%s | early stopping at epoch %d (val loss not improving)",
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
        model_path = model_output_dir / f"ablation_weights_{experiment_name.lower()}.pt"
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
