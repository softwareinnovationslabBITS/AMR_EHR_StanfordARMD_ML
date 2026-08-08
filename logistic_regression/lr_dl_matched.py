#!/usr/bin/env python3

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
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
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
    roc_curve,
)
from sklearn.calibration import calibration_curve


warnings.filterwarnings("ignore", category=FutureWarning)


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

start_total = time.time()

bundle = joblib.load(bundle_path)


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

print(f"Iterations used: {model.n_iter_[0]}")


# ======================================================================
# 14. GENERATE VALIDATION AND TEST PROBABILITIES
# ======================================================================

print("\nGenerating validation probabilities...")

val_prob = model.predict_proba(X_val_lr)[:, 1]

print("Generating test probabilities...")

test_prob = model.predict_proba(X_test_lr)[:, 1]


# ======================================================================
# 15. THRESHOLD OPTIMIZATION ON VALIDATION SET ONLY
# ======================================================================

print("\n" + "=" * 80)
print("SELECTING THRESHOLD ON VALIDATION SET")
print("=" * 80)


def select_threshold_by_mcc(y_true, probabilities):

    thresholds = np.linspace(
        0.0,
        1.0,
        N_THRESHOLD_CANDIDATES,
    )

    best_threshold = 0.5
    best_mcc = -np.inf

    rows = []

    for threshold in thresholds:

        pred = (probabilities >= threshold).astype(np.int8)

        # Avoid degenerate one-class predictions causing meaningless values.
        if len(np.unique(pred)) < 2:
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(y_true, pred)

        rows.append(
            {
                "threshold": threshold,
                "mcc": mcc,
            }
        )

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    threshold_df = pd.DataFrame(rows)

    return best_threshold, best_mcc, threshold_df


best_threshold, best_val_mcc, threshold_df = select_threshold_by_mcc(
    y_val,
    val_prob,
)

print(f"Best validation threshold : {best_threshold:.4f}")
print(f"Validation MCC            : {best_val_mcc:.6f}")

threshold_df.to_csv(
    OUTPUT_DIR / "validation_threshold_search.csv",
    index=False,
)


# ======================================================================
# 16. METRIC FUNCTION
# ======================================================================

def calculate_metrics(y_true, probabilities, threshold):

    pred = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),

        "accuracy": accuracy_score(
            y_true,
            pred,
        ),

        "precision": precision_score(
            y_true,
            pred,
            zero_division=0,
        ),

        "recall_sensitivity": recall_score(
            y_true,
            pred,
            zero_division=0,
        ),

        "specificity": specificity,

        "npv": npv,

        "f1": f1_score(
            y_true,
            pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            pred,
        ),

        "mcc": matthews_corrcoef(
            y_true,
            pred,
        ),

        "cohen_kappa": cohen_kappa_score(
            y_true,
            pred,
        ),

        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


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

pd.DataFrame([metrics_row]).to_csv(
    OUTPUT_DIR / "logistic_regression_test_metrics.csv",
    index=False,
)


pd.DataFrame(
    [
        {
            "threshold_type": "validation_MCC_optimized",
            **test_metrics,
        },
        {
            "threshold_type": "default_0.50",
            **default_metrics,
        },
    ]
).to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False,
)


# ======================================================================
# 21. SAVE PREDICTIONS
# ======================================================================

prediction_df = pd.DataFrame(
    {
        "y_true": y_test,
        "probability_resistant": test_prob,
        "predicted_class": (
            test_prob >= best_threshold
        ).astype(np.int8),
    }
)

prediction_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False,
)


# ======================================================================
# 22. BOOTSTRAP 95% CONFIDENCE INTERVALS
# ======================================================================

print("\n" + "=" * 80)
print(
    f"BOOTSTRAP TEST CONFIDENCE INTERVALS "
    f"({N_BOOTSTRAPS:,} replicates)"
)
print("=" * 80)


def stratified_bootstrap_indices(y, rng):

    idx_0 = np.flatnonzero(y == 0)
    idx_1 = np.flatnonzero(y == 1)

    boot_0 = rng.choice(
        idx_0,
        size=len(idx_0),
        replace=True,
    )

    boot_1 = rng.choice(
        idx_1,
        size=len(idx_1),
        replace=True,
    )

    idx = np.concatenate(
        [boot_0, boot_1]
    )

    rng.shuffle(idx)

    return idx


BOOTSTRAP_METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "npv",
    "f1",
    "balanced_accuracy",
    "mcc",
    "cohen_kappa",
    "brier_score",
]


bootstrap_results = {
    metric: []
    for metric in BOOTSTRAP_METRICS
}

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_start = time.time()

for b in range(N_BOOTSTRAPS):

    idx = stratified_bootstrap_indices(
        y_test,
        rng,
    )

    y_b = y_test[idx]
    p_b = test_prob[idx]

    m = calculate_metrics(
        y_b,
        p_b,
        best_threshold,
    )

    for metric in BOOTSTRAP_METRICS:
        bootstrap_results[metric].append(
            m[metric]
        )

    if (
        (b + 1) % 100 == 0
        or b == 0
    ):
        print(
            f"Bootstrap "
            f"{b + 1:,}/{N_BOOTSTRAPS:,}"
        )


bootstrap_rows = []

for metric in BOOTSTRAP_METRICS:

    values = np.asarray(
        bootstrap_results[metric]
    )

    point_estimate = test_metrics[metric]

    lower = np.percentile(
        values,
        2.5,
    )

    upper = np.percentile(
        values,
        97.5,
    )

    bootstrap_rows.append(
        {
            "metric": metric,
            "estimate": point_estimate,
            "ci_lower_95": lower,
            "ci_upper_95": upper,
        }
    )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df.to_csv(
    OUTPUT_DIR
    / "logistic_regression_bootstrap_95CI.csv",
    index=False,
)

print("\nBootstrap 95% CIs")
print(
    bootstrap_df.to_string(
        index=False
    )
)

print(
    f"\nBootstrap completed in "
    f"{(time.time() - bootstrap_start)/60:.2f} minutes"
)


# ======================================================================
# 23. ROC CURVE
# ======================================================================

fpr, tpr, _ = roc_curve(
    y_test,
    test_prob,
)

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

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ======================================================================
# 24. PRECISION-RECALL CURVE
# ======================================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    test_prob,
)

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
    OUTPUT_DIR / "precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ======================================================================
# 25. CONFUSION MATRIX
# ======================================================================

test_pred = (
    test_prob >= best_threshold
).astype(np.int8)

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1],
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.xticks(
    [0, 1],
    ["Susceptible", "Resistant"],
)

plt.yticks(
    [0, 1],
    ["Susceptible", "Resistant"],
)

plt.xlabel("Predicted")
plt.ylabel("Observed")

plt.title(
    "Logistic Regression Confusion Matrix\n"
    f"Validation-selected threshold = "
    f"{best_threshold:.3f}"
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

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ======================================================================
# 26. CALIBRATION CURVE
# ======================================================================

fraction_positive, mean_predicted = calibration_curve(
    y_test,
    test_prob,
    n_bins=N_CALIBRATION_BINS,
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

plt.title(
    "Logistic Regression Calibration"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "calibration_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ======================================================================
# 27. VALIDATION THRESHOLD CURVE
# ======================================================================

plt.figure(figsize=(7, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["mcc"],
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=(
        f"Selected threshold="
        f"{best_threshold:.3f}"
    ),
)

plt.xlabel("Classification threshold")
plt.ylabel("Validation MCC")
plt.title("Validation Threshold Selection")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "validation_threshold_mcc.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ======================================================================
# 28. COEFFICIENT IMPORTANCE
# ======================================================================

coefficients = model.coef_[0]

coef_df = pd.DataFrame(
    {
        "feature": expanded_feature_names,
        "coefficient": coefficients,
    }
)

coef_df["absolute_coefficient"] = np.abs(
    coef_df["coefficient"]
)

coef_df = coef_df.sort_values(
    "absolute_coefficient",
    ascending=False,
)

coef_df.to_csv(
    OUTPUT_DIR / "all_logistic_coefficients.csv",
    index=False,
)


top_coef = coef_df.head(
    TOP_N_COEFFICIENTS
).copy()

# Reverse so largest appears at top in horizontal plot.
top_coef = top_coef.iloc[::-1]

plt.figure(
    figsize=(10, 10)
)

plt.barh(
    top_coef["feature"],
    top_coef["coefficient"],
)

plt.xlabel("Logistic regression coefficient")
plt.ylabel("Feature")
plt.title(
    f"Top {TOP_N_COEFFICIENTS} Logistic Regression Coefficients"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_logistic_coefficients.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ======================================================================
# 29. SAVE MODEL AND ENCODER
# ======================================================================

joblib.dump(
    model,
    OUTPUT_DIR / "logistic_regression_model.joblib",
)

joblib.dump(
    encoder,
    OUTPUT_DIR / "onehot_encoder.joblib",
)


# ======================================================================
# 30. SAVE CONFIGURATION
# ======================================================================

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


with open(
    OUTPUT_DIR / "run_config.json",
    "w",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


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
