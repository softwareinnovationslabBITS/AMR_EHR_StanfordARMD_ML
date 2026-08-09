#!/usr/bin/env python3
# Source: /AMR_Stanford/DL_codes/amr_project/xgb_dl_feature_matched_project/04_logistic_regression_feature_matched.py

# ======================================================================
# FEATURE-MATCHED LOGISTIC REGRESSION BENCHMARK
# ======================================================================
#
# Purpose
# -------
# Benchmark logistic regression against XGBoost and TabTransformer using:
#
#   - EXACT same observations
#   - EXACT same S/R outcome
#   - EXACT same train/validation/test splits
#   - EXACT same underlying 631 features
#
# Model-specific handling:
#
#   Categorical features:
#       One-hot encoded using training data only
#
#   Continuous features:
#       Retained exactly as stored in the DL bundle
#       (already standardized in the original DL workflow)
#
#   Binary features:
#       Retained as 0/1
#
# Outcome:
#       Susceptible = 0
#       Resistant   = 1
#
# Threshold:
#       Selected on VALIDATION data by maximum MCC
#       Test set is NOT used for threshold selection
#
# Outputs:
#       metrics
#       bootstrap confidence intervals
#       ROC curve
#       precision-recall curve
#       calibration curve
#       confusion matrix
#       coefficient importance
#       predictions
#
# ======================================================================


# ======================================================================
# 1. IMPORTS
# ======================================================================

import gc
import json
import logging
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# #migrate: split plotting, metrics, and I/O helpers into private modules
from lr_io import (
    save_bootstrap_results,
    save_model_and_encoder,
    save_predictions,
    save_run_config,
    save_test_metrics_row,
    save_threshold_comparison,
)
from lr_metrics import (
    calculate_metrics,
    run_stratified_bootstrap,
    select_threshold_by_mcc,
    summarize_bootstrap_ci,
)
from lr_plotting import generate_all_plots

warnings.filterwarnings("ignore", category=FutureWarning)

# #migrate: configure timestamped progress logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# 2. CONFIGURATION
# ======================================================================

# #migrate: load seed and bundle path from the single config file
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
RANDOM_SEED = _CFG.get('seed', 42)
BUNDLE_PATH = resolve_path(_CFG.get('tabtransformer', {}).get('bundle_path', 'dataset/amr_analysis_bundle.joblib'))

# #migrate: add shared dataset/ folder to bundle search path
BUNDLE_CANDIDATES = [
    Path("amr_analysis_bundle.joblib"),
    Path("../amr_analysis_bundle.joblib"),
    Path("../dataset/amr_analysis_bundle.joblib"),
    BUNDLE_PATH,
]

# #migrate: output directory from the single config file
_PATHS_CFG = _CFG.get('paths', {})
OUTPUT_DIR = Path(str(resolve_path(_PATHS_CFG.get('logistic_regression_output_dir', 'logistic_regression/logistic_regression_dl_matched_outputs'))))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logistic regression configuration
#
# saga:
#   - works well with sparse matrices
#   - appropriate for large datasets
#   - supports L2 regularization
#
# class_weight="balanced":
#   gives the resistant class additional weight without changing the
#   physical training-set class distribution.
#
# This is the logistic-regression analogue of cost-sensitive learning.
LR_C = 1.0
LR_MAX_ITER = 200
LR_TOL = 1e-4
LR_CLASS_WEIGHT = "balanced"

# Threshold optimization
THRESHOLD_METRIC = "MCC"
N_THRESHOLD_CANDIDATES = 1001

# Bootstrap
N_BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 42

# Calibration
N_CALIBRATION_BINS = 10

# Number of coefficients displayed
TOP_N_COEFFICIENTS = 30


# ======================================================================
# 3. FIND AND LOAD ANALYSIS BUNDLE
# ======================================================================

bundle_path = None

for candidate in BUNDLE_CANDIDATES:
    if candidate.exists():
        bundle_path = candidate
        break

if bundle_path is None:
    raise FileNotFoundError(
        "\nCould not find amr_analysis_bundle.joblib.\n"
        "Checked:\n"
        + "\n".join(str(x.resolve()) for x in BUNDLE_CANDIDATES)
    )

print("=" * 80)
print("FEATURE-MATCHED LOGISTIC REGRESSION")
print("=" * 80)
print(f"\nLoading DL analysis bundle:")
print(bundle_path.resolve())
logger.info("Loading analysis bundle from %s", bundle_path.resolve())

start_total = time.time()

bundle = joblib.load(bundle_path)
logger.info("Analysis bundle loaded")


# ======================================================================
# 4. LOAD EXACT SAVED MATRICES AND FEATURE DEFINITIONS
# ======================================================================

X_train = np.asarray(bundle["X_train"])
X_val   = np.asarray(bundle["X_val"])
X_test  = np.asarray(bundle["X_test"])

y_train = np.asarray(bundle["y_train"]).ravel().astype(np.int8)
y_val   = np.asarray(bundle["y_val"]).ravel().astype(np.int8)
y_test  = np.asarray(bundle["y_test"]).ravel().astype(np.int8)

CAT_FEATURES = list(bundle["CAT_FEATURES"])
CONT_FEATURES = list(bundle["CONT_FEATURES"])
BINARY_FEATURES = list(bundle["BINARY_FEATURES"])
ALL_FEATURES = list(bundle["ALL_FEATURES"])

cat_idx = np.asarray(bundle["cat_idx"], dtype=int)
cont_idx = np.asarray(bundle["cont_idx"], dtype=int)
bin_idx = np.asarray(bundle["bin_idx"], dtype=int)


print("\nDataset dimensions")
print("-" * 80)

print(f"Training   : {len(y_train):,}")
print(f"Validation : {len(y_val):,}")
print(f"Test       : {len(y_test):,}")

print(f"\nTotal underlying features : {len(ALL_FEATURES):,}")
print(f"Categorical               : {len(CAT_FEATURES):,}")
print(f"Continuous                : {len(CONT_FEATURES):,}")
print(f"Binary                    : {len(BINARY_FEATURES):,}")
logger.info(
    "Dataset loaded | train=%d val=%d test=%d | features=%d (cat=%d cont=%d bin=%d)",
    len(y_train),
    len(y_val),
    len(y_test),
    len(ALL_FEATURES),
    len(CAT_FEATURES),
    len(CONT_FEATURES),
    len(BINARY_FEATURES),
)


# ======================================================================
# 5. VERIFY OUTCOME DISTRIBUTION
# ======================================================================

def print_class_distribution(name, y):

    s = int(np.sum(y == 0))
    r = int(np.sum(y == 1))

    print(f"\n{name}")
    print(f"  S = {s:,}")
    print(f"  R = {r:,}")
    print(f"  S:R = {s/r:.3f}:1")
    print(f"  Resistance prevalence = {100*r/len(y):.2f}%")


print_class_distribution("TRAIN", y_train)
print_class_distribution("VALIDATION", y_val)
print_class_distribution("TEST", y_test)


# ======================================================================
# 6. SPLIT FEATURE TYPES
# ======================================================================

print("\n" + "=" * 80)
print("PREPARING FEATURE TYPES")
print("=" * 80)


# ----------------------------------------------------------------------
# Categorical
# ----------------------------------------------------------------------

# Integer categorical IDs from TabTransformer bundle.
#
# Convert to int64 because OneHotEncoder should treat each integer as a
# category rather than a continuous measurement.

X_train_cat = X_train[:, cat_idx].astype(np.int64, copy=False)
X_val_cat   = X_val[:, cat_idx].astype(np.int64, copy=False)
X_test_cat  = X_test[:, cat_idx].astype(np.int64, copy=False)


# ----------------------------------------------------------------------
# Continuous
# ----------------------------------------------------------------------

# These values are retained exactly from the DL bundle.
# The original DL preprocessing already standardized continuous features.

X_train_cont = X_train[:, cont_idx].astype(np.float32, copy=False)
X_val_cont   = X_val[:, cont_idx].astype(np.float32, copy=False)
X_test_cont  = X_test[:, cont_idx].astype(np.float32, copy=False)


# ----------------------------------------------------------------------
# Binary
# ----------------------------------------------------------------------

X_train_bin = X_train[:, bin_idx].astype(np.float32, copy=False)
X_val_bin   = X_val[:, bin_idx].astype(np.float32, copy=False)
X_test_bin  = X_test[:, bin_idx].astype(np.float32, copy=False)


# ======================================================================
# 7. FIT ONE-HOT ENCODER ON TRAINING DATA ONLY
# ======================================================================

print("\nFitting OneHotEncoder on TRAINING categorical data only...")

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
    dtype=np.float32,
)

t0 = time.time()

encoder.fit(X_train_cat)

print(
    f"OneHotEncoder fitted in "
    f"{(time.time() - t0)/60:.2f} minutes"
)
logger.info("One-hot encoding fitted on training categorical data")


# ======================================================================
# 8. TRANSFORM CATEGORICAL FEATURES
# ======================================================================

print("\nTransforming categorical variables...")

X_train_cat_ohe = encoder.transform(X_train_cat)
X_val_cat_ohe   = encoder.transform(X_val_cat)
X_test_cat_ohe  = encoder.transform(X_test_cat)

print(
    f"One-hot categorical columns: "
    f"{X_train_cat_ohe.shape[1]:,}"
)


# ======================================================================
# 9. CONVERT CONTINUOUS/BINARY MATRICES TO SPARSE FORMAT
# ======================================================================

print("\nConverting continuous and binary matrices to sparse format...")

X_train_cont_sparse = sparse.csr_matrix(X_train_cont)
X_val_cont_sparse   = sparse.csr_matrix(X_val_cont)
X_test_cont_sparse  = sparse.csr_matrix(X_test_cont)

X_train_bin_sparse = sparse.csr_matrix(X_train_bin)
X_val_bin_sparse   = sparse.csr_matrix(X_val_bin)
X_test_bin_sparse  = sparse.csr_matrix(X_test_bin)


# ======================================================================
# 10. COMBINE FEATURE MATRICES
# ======================================================================

print("\nCombining feature matrices...")

X_train_lr = sparse.hstack(
    [
        X_train_cat_ohe,
        X_train_cont_sparse,
        X_train_bin_sparse,
    ],
    format="csr",
    dtype=np.float32,
)

X_val_lr = sparse.hstack(
    [
        X_val_cat_ohe,
        X_val_cont_sparse,
        X_val_bin_sparse,
    ],
    format="csr",
    dtype=np.float32,
)

X_test_lr = sparse.hstack(
    [
        X_test_cat_ohe,
        X_test_cont_sparse,
        X_test_bin_sparse,
    ],
    format="csr",
    dtype=np.float32,
)


print("\nFinal LR matrices")
print("-" * 80)
print(f"Train : {X_train_lr.shape}")
print(f"Val   : {X_val_lr.shape}")
print(f"Test  : {X_test_lr.shape}")
logger.info(
    "Final LR matrices | train=%s val=%s test=%s",
    X_train_lr.shape,
    X_val_lr.shape,
    X_test_lr.shape,
)


# ======================================================================
# 11. FREE UNNECESSARY INTERMEDIATE ARRAYS
# ======================================================================

del X_train_cat, X_val_cat, X_test_cat
del X_train_cont, X_val_cont, X_test_cont
del X_train_bin, X_val_bin, X_test_bin

del X_train_cont_sparse
del X_val_cont_sparse
del X_test_cont_sparse

del X_train_bin_sparse
del X_val_bin_sparse
del X_test_bin_sparse

gc.collect()


# ======================================================================
# 12. CREATE EXPANDED FEATURE NAMES
# ======================================================================

categorical_names = encoder.get_feature_names_out(CAT_FEATURES).tolist()

expanded_feature_names = (
    categorical_names
    + CONT_FEATURES
    + BINARY_FEATURES
)

assert len(expanded_feature_names) == X_train_lr.shape[1]

print(
    f"\nExpanded LR design matrix features: "
    f"{len(expanded_feature_names):,}"
)
logger.info("Expanded LR design matrix features: %d", len(expanded_feature_names))


# ======================================================================
# 13. TRAIN LOGISTIC REGRESSION
# ======================================================================

print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 80)

print(f"C                  : {LR_C}")
print(f"Penalty            : L2")
print(f"Solver             : saga")
print(f"Class weight       : {LR_CLASS_WEIGHT}")
print(f"Maximum iterations : {LR_MAX_ITER}")
print(f"Tolerance          : {LR_TOL}")

model = LogisticRegression(
    penalty="l2",
    C=LR_C,
    solver="saga",
    class_weight=LR_CLASS_WEIGHT,
    max_iter=LR_MAX_ITER,
    tol=LR_TOL,
    random_state=RANDOM_SEED,
    verbose=1,
)

train_start = time.time()

model.fit(X_train_lr, y_train)

training_minutes = (time.time() - train_start) / 60

print(
    f"\nLogistic regression training completed in "
    f"{training_minutes:.2f} minutes"
)
logger.info("Logistic regression training completed in %.2f minutes", training_minutes)

print(f"Iterations used: {model.n_iter_[0]}")
logger.info("Iterations used: %d", model.n_iter_[0])


# ======================================================================
# 14. GENERATE VALIDATION AND TEST PROBABILITIES
# ======================================================================

print("\nGenerating validation probabilities...")
logger.info("Generating validation probabilities")

val_prob = model.predict_proba(X_val_lr)[:, 1]

print("Generating test probabilities...")
logger.info("Generating test probabilities")

test_prob = model.predict_proba(X_test_lr)[:, 1]


# ======================================================================
# 15. THRESHOLD OPTIMIZATION ON VALIDATION SET ONLY
# ======================================================================

print("\n" + "=" * 80)
print("SELECTING THRESHOLD ON VALIDATION SET")
print("=" * 80)

logger.info("Selecting threshold on validation set by maximum MCC")

best_threshold, best_val_mcc, threshold_df = select_threshold_by_mcc(
    y_val,
    val_prob,
    n_threshold_candidates=N_THRESHOLD_CANDIDATES,
)

print(f"Best validation threshold : {best_threshold:.4f}")
print(f"Validation MCC            : {best_val_mcc:.6f}")

threshold_df.to_csv(
    OUTPUT_DIR / "validation_threshold_search.csv",
    index=False,
)
logger.info(
    "Validation threshold selected: %.4f (MCC=%.6f)",
    best_threshold,
    best_val_mcc,
)


# ======================================================================
# 16. METRICS
# ======================================================================

logger.info("Calculating validation and test metrics")


# ======================================================================
# 17. VALIDATION METRICS
# ======================================================================

validation_metrics = calculate_metrics(
    y_val,
    val_prob,
    best_threshold,
)


# ======================================================================
# 18. FINAL TEST METRICS
# ======================================================================

test_metrics = calculate_metrics(
    y_test,
    test_prob,
    best_threshold,
)


print("\n" + "=" * 80)
print("FINAL TEST PERFORMANCE")
print("=" * 80)

for key, value in test_metrics.items():

    if isinstance(value, float):
        print(f"{key:25s}: {value:.6f}")
    else:
        print(f"{key:25s}: {value:,}")


# ======================================================================
# 19. ALSO CALCULATE DEFAULT 0.50 THRESHOLD
# ======================================================================

default_metrics = calculate_metrics(
    y_test,
    test_prob,
    0.50,
)


# ======================================================================
# 20. SAVE METRICS
# ======================================================================

metrics_row = {
    "model": "Logistic Regression",
    "feature_representation": "DL_feature_matched",
    "underlying_features": len(ALL_FEATURES),
    "expanded_lr_features": X_train_lr.shape[1],
    "n_train": len(y_train),
    "n_validation": len(y_val),
    "n_test": len(y_test),
    "class_weight": str(LR_CLASS_WEIGHT),
    "C": LR_C,
    "solver": "saga",
    "training_minutes": training_minutes,
    "iterations": int(model.n_iter_[0]),
    **{
        f"test_{k}": v
        for k, v in test_metrics.items()
    },
}

save_test_metrics_row(metrics_row, OUTPUT_DIR)
save_threshold_comparison(test_metrics, default_metrics, OUTPUT_DIR)


# ======================================================================
# 21. SAVE PREDICTIONS
# ======================================================================

save_predictions(y_test, test_prob, best_threshold, OUTPUT_DIR)


# ======================================================================
# 22. BOOTSTRAP 95% CONFIDENCE INTERVALS
# ======================================================================

print("\n" + "=" * 80)
print(
    f"BOOTSTRAP TEST CONFIDENCE INTERVALS "
    f"({N_BOOTSTRAPS:,} replicates)"
)
print("=" * 80)


logger.info(
    "Starting bootstrap confidence intervals (%d replicates)",
    N_BOOTSTRAPS,
)

# #migrate: progress logging every 250 iterations is handled in lr_metrics.py
bootstrap_results = run_stratified_bootstrap(
    y_test=y_test,
    test_prob=test_prob,
    best_threshold=best_threshold,
    n_bootstraps=N_BOOTSTRAPS,
    bootstrap_seed=BOOTSTRAP_SEED,
    progress_every=250,
)

bootstrap_df = summarize_bootstrap_ci(
    bootstrap_results,
    test_metrics,
    confidence_level=0.95,
)

save_bootstrap_results(bootstrap_df, OUTPUT_DIR)

print("\nBootstrap 95% CIs")
print(bootstrap_df.to_string(index=False))
logger.info("Bootstrap confidence intervals completed")


# ======================================================================
# 23-28. PLOTS
# ======================================================================

generate_all_plots(
    y_test=y_test,
    test_prob=test_prob,
    test_metrics=test_metrics,
    threshold_df=threshold_df,
    best_threshold=best_threshold,
    model=model,
    expanded_feature_names=expanded_feature_names,
    n_calibration_bins=N_CALIBRATION_BINS,
    top_n_coefficients=TOP_N_COEFFICIENTS,
    output_dir=OUTPUT_DIR,
)


# ======================================================================
# 29-30. SAVE MODEL, ENCODER, AND CONFIGURATION
# ======================================================================

save_model_and_encoder(model, encoder, OUTPUT_DIR)

config = {
    "random_seed": RANDOM_SEED,
    "bundle_path": str(bundle_path.resolve()),
    "n_underlying_features": len(ALL_FEATURES),
    "n_categorical_features": len(CAT_FEATURES),
    "n_continuous_features": len(CONT_FEATURES),
    "n_binary_features": len(BINARY_FEATURES),
    "n_expanded_lr_features": X_train_lr.shape[1],
    "categorical_features": CAT_FEATURES,
    "continuous_features": CONT_FEATURES,
    "class_weight": LR_CLASS_WEIGHT,
    "C": LR_C,
    "solver": "saga",
    "penalty": "l2",
    "max_iter": LR_MAX_ITER,
    "tolerance": LR_TOL,
    "threshold_selection": "maximum MCC on validation set",
    "selected_threshold": float(best_threshold),
    "bootstrap_replicates": N_BOOTSTRAPS,
}

save_run_config(config, OUTPUT_DIR)


# ======================================================================
# 31. FINAL SUMMARY
# ======================================================================

total_minutes = (
    time.time() - start_total
) / 60

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION COMPLETE")
print("=" * 80)

print(
    f"\nUnderlying DL features : "
    f"{len(ALL_FEATURES):,}"
)

print(
    f"Expanded LR features   : "
    f"{X_train_lr.shape[1]:,}"
)

print(
    f"Validation threshold   : "
    f"{best_threshold:.4f}"
)

print(
    f"\nTest ROC-AUC           : "
    f"{test_metrics['roc_auc']:.6f}"
)

print(
    f"Test PR-AUC            : "
    f"{test_metrics['pr_auc']:.6f}"
)

print(
    f"Test balanced accuracy : "
    f"{test_metrics['balanced_accuracy']:.6f}"
)

print(
    f"Test MCC               : "
    f"{test_metrics['mcc']:.6f}"
)

print(
    f"Test F1                : "
    f"{test_metrics['f1']:.6f}"
)

print(
    f"Test Brier score       : "
    f"{test_metrics['brier_score']:.6f}"
)

print(
    f"\nTraining time          : "
    f"{training_minutes:.2f} min"
)

print(
    f"Total script time      : "
    f"{total_minutes:.2f} min"
)

print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nDone.")
