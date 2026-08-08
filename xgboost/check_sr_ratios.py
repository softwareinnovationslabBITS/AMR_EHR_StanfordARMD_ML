#!/usr/bin/env python3

import joblib
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BUNDLE_PATH = Path("amr_analysis_bundle.joblib")

# Same settings used in the feature-matched XGBoost workflow
SMOTENC_SAMPLING_STRATEGY = 0.50
UNDERSAMPLING_STRATEGY = 1.0


# ============================================================
# HELPERS
# ============================================================

def summarize_split(name, y):
    y = np.asarray(y).ravel()

    s = int(np.sum(y == 0))
    r = int(np.sum(y == 1))
    total = len(y)

    sr_ratio = s / r if r > 0 else np.nan
    resistance_pct = 100 * r / total if total > 0 else np.nan

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(f"Total observations       : {total:,}")
    print(f"Susceptible (S)          : {s:,}")
    print(f"Resistant (R)            : {r:,}")
    print(f"S:R ratio                : {sr_ratio:.3f}:1")
    print(f"Resistance prevalence    : {resistance_pct:.2f}%")

    return {
        "total": total,
        "S": s,
        "R": r,
        "S_R_ratio": sr_ratio,
        "resistance_pct": resistance_pct,
    }


# ============================================================
# LOAD BUNDLE
# ============================================================

if not BUNDLE_PATH.exists():
    raise FileNotFoundError(
        f"Could not find: {BUNDLE_PATH.resolve()}"
    )

print("Loading:", BUNDLE_PATH)

bundle = joblib.load(BUNDLE_PATH)

y_train = np.asarray(bundle["y_train"]).ravel()
y_val = np.asarray(bundle["y_val"]).ravel()
y_test = np.asarray(bundle["y_test"]).ravel()


# ============================================================
# ORIGINAL SPLITS
# ============================================================

train_info = summarize_split(
    "ORIGINAL TRAINING SET",
    y_train
)

val_info = summarize_split(
    "VALIDATION SET",
    y_val
)

test_info = summarize_split(
    "TEST SET",
    y_test
)


# ============================================================
# FULL COMBINED DATASET
# ============================================================

y_all = np.concatenate([
    y_train,
    y_val,
    y_test
])

all_info = summarize_split(
    "FULL COMBINED DATASET",
    y_all
)


# ============================================================
# CLASS-WEIGHTED XGBOOST
# ============================================================

S_train = train_info["S"]
R_train = train_info["R"]

scale_pos_weight = S_train / R_train

print("\n" + "=" * 70)
print("CLASS-WEIGHTED XGBOOST")
print("=" * 70)
print("Training data itself is NOT resampled.")
print(f"Original training S      : {S_train:,}")
print(f"Original training R      : {R_train:,}")
print(f"Original S:R ratio       : {S_train / R_train:.3f}:1")
print(f"scale_pos_weight         : {scale_pos_weight:.6f}")


# ============================================================
# SMOTENC
# ============================================================

# For imbalanced-learn oversampling:
# sampling_strategy = minority / majority after resampling
#
# Here:
# R_after / S_after = 0.50
# Therefore:
# S:R = 2:1

smote_S = S_train
smote_R = int(round(
    SMOTENC_SAMPLING_STRATEGY * smote_S
))

synthetic_R_added = smote_R - R_train
smote_total = smote_S + smote_R

print("\n" + "=" * 70)
print("SMOTENC OVERSAMPLING")
print("=" * 70)

print("\nBEFORE SMOTENC")
print(f"S                       : {S_train:,}")
print(f"R                       : {R_train:,}")
print(f"S:R ratio               : {S_train / R_train:.3f}:1")
print(
    f"Resistance prevalence   : "
    f"{100 * R_train / (S_train + R_train):.2f}%"
)

print("\nAFTER SMOTENC")
print(
    f"sampling_strategy        : "
    f"{SMOTENC_SAMPLING_STRATEGY}"
)
print(f"S                       : {smote_S:,}")
print(f"R                       : {smote_R:,}")
print(f"S:R ratio               : {smote_S / smote_R:.3f}:1")
print(
    f"Resistance prevalence   : "
    f"{100 * smote_R / smote_total:.2f}%"
)
print(f"Synthetic R rows added  : {synthetic_R_added:,}")
print(f"Total training rows     : {smote_total:,}")


# ============================================================
# RANDOM UNDERSAMPLING
# ============================================================

# For RandomUnderSampler:
# sampling_strategy = minority / majority after undersampling
#
# Here:
# R_after / S_after = 1.0
# Therefore:
# S:R = 1:1

under_R = R_train
under_S = int(round(
    under_R / UNDERSAMPLING_STRATEGY
))

S_removed = S_train - under_S
under_total = under_S + under_R

print("\n" + "=" * 70)
print("RANDOM UNDERSAMPLING")
print("=" * 70)

print("\nBEFORE UNDERSAMPLING")
print(f"S                       : {S_train:,}")
print(f"R                       : {R_train:,}")
print(f"S:R ratio               : {S_train / R_train:.3f}:1")
print(
    f"Resistance prevalence   : "
    f"{100 * R_train / (S_train + R_train):.2f}%"
)

print("\nAFTER UNDERSAMPLING")
print(
    f"sampling_strategy        : "
    f"{UNDERSAMPLING_STRATEGY}"
)
print(f"S                       : {under_S:,}")
print(f"R                       : {under_R:,}")
print(f"S:R ratio               : {under_S / under_R:.3f}:1")
print(
    f"Resistance prevalence   : "
    f"{100 * under_R / under_total:.2f}%"
)
print(f"Susceptible rows removed: {S_removed:,}")
print(f"Total training rows     : {under_total:,}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Original train       S:R = "
    f"{S_train / R_train:.3f}:1"
)

print(
    f"Validation           S:R = "
    f"{val_info['S_R_ratio']:.3f}:1"
)

print(
    f"Test                 S:R = "
    f"{test_info['S_R_ratio']:.3f}:1"
)

print(
    f"Full dataset         S:R = "
    f"{all_info['S_R_ratio']:.3f}:1"
)

print(
    f"Class weighted       S:R = "
    f"{S_train / R_train:.3f}:1 "
    f"(unchanged; scale_pos_weight="
    f"{scale_pos_weight:.3f})"
)

print(
    f"SMOTENC              S:R = "
    f"{smote_S / smote_R:.3f}:1"
)

print(
    f"Undersampling        S:R = "
    f"{under_S / under_R:.3f}:1"
)

print("\nDone.")
