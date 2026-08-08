"""
utils.py
--------
Shared helpers used by every technique script (01-07).
Keeping this in one place means every method is scored identically,
so the final comparison table is actually apples-to-apples.

Also handles per-method MODEL SAVING, mirroring your original code's
MODEL_DIR / MODEL_PATH / META_PATH pattern — each script gets its own
./saved_models/<method_name>/ folder containing:
  xgb_model.ubj   - the trained XGBoost model (native format)
  meta.json       - hyperparameters, scale_pos_weight, threshold, etc.
  feature_importance.csv

The preprocessor itself isn't re-saved per script (it's shared and already
cached once by 00_common.py at ./cache/preprocessor.joblib) — only the
model and that script's own settings/results are saved per technique.
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
)

RESULTS_DIR = "./results"
MODELS_DIR = "./saved_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
RESULTS_LOG = os.path.join(RESULTS_DIR, "comparison_log.json")


def compute_metrics(y_test, y_pred, y_prob):
    """One canonical metric block, reused by every script."""
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
    }


def class_ratio(y):
    """Returns (n_susceptible, n_resistant, S:R ratio) for a label array."""
    y = np.asarray(y)
    n_neg = int(np.sum(y == 0))  # Susceptible
    n_pos = int(np.sum(y == 1))  # Resistant
    ratio = n_neg / n_pos if n_pos > 0 else float("nan")
    return n_neg, n_pos, ratio


def print_ratio_change(name, y_before, y_after):
    nb_neg, nb_pos, nb_ratio = class_ratio(y_before)
    na_neg, na_pos, na_ratio = class_ratio(y_after)
    print(f"\n[RATIO] {name}")
    print(f"        Before -> S={nb_neg:,}  R={nb_pos:,}  S:R = {nb_ratio:.2f}:1")
    print(f"        After  -> S={na_neg:,}  R={na_pos:,}  S:R = {na_ratio:.2f}:1")
    return {
        "before": {"S": nb_neg, "R": nb_pos, "ratio": nb_ratio},
        "after": {"S": na_neg, "R": na_pos, "ratio": na_ratio},
    }


def print_report(method_name, metrics, extra=None, elapsed=None):
    print("\n" + "=" * 60)
    print(f"  RESULTS — {method_name}")
    print("=" * 60)
    print(f"[METRIC] ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"[METRIC] PR-AUC:             {metrics['pr_auc']:.4f}")
    print(f"[METRIC] Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"[METRIC] MCC:                {metrics['mcc']:.4f}")
    print(f"[METRIC] Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
    if extra:
        for k, v in extra.items():
            print(f"[INFO]   {k}: {v}")
    if elapsed is not None:
        print(f"[TIME]   {elapsed:.1f}s")
    print("=" * 60)


def print_classification_report(y_true, y_pred):
    """Mirrors your original Phase 11 '[REPORT] Classification Report:' block."""
    print("\n[REPORT] Classification Report:")
    print(classification_report(y_true, y_pred))


def get_feature_names(preprocessor, cat_cols, num_cols):
    """Returns the flat, ordered list of feature names matching the columns
    of the preprocessor's output matrix: one-hot category names first, then
    passthrough numeric names — same order ColumnTransformer concatenates
    them in. Shared by print_top_features() and every visualization script
    so the name <-> column mapping is computed identically everywhere."""
    cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat_cols)
    return list(cat_names) + list(num_cols)


def print_top_features(model, preprocessor, cat_cols, num_cols, top_n=15):
    """Mirrors your original Phase 11 '[INFO] Top 15 Predictors:' block.
    Returns the full feature importance DataFrame (not just top_n) so callers
    can save the complete table to disk even though only top_n is printed."""
    try:
        all_names = get_feature_names(preprocessor, cat_cols, num_cols)
        importance = model.feature_importances_

        if len(all_names) != len(importance):
            print(f"[WARN] Feature name count ({len(all_names)}) != importance count "
                  f"({len(importance)}) — skipping feature importance table.")
            return None

        feat_imp = pd.DataFrame({'Feature': all_names, 'Importance': importance}) \
            .sort_values(by='Importance', ascending=False) \
            .reset_index(drop=True)

        print(f"\n[INFO] Top {top_n} Predictors:")
        print("-" * 50)
        print(feat_imp.head(top_n).to_string(index=False))
        return feat_imp
    except Exception as e:
        print(f"[ERROR] Could not extract feature names automatically: {e}")
        return None


def save_model_artifacts(method_name, model, meta, feat_imp_df=None):
    """Saves the trained model + its metadata to ./saved_models/<method_name>/,
    mirroring your original MODEL_PATH / META_PATH save pattern. Works for any
    sklearn-API XGBoost model (XGBClassifier).

    Returns the directory path the artifacts were saved to.
    """
    method_dir = os.path.join(MODELS_DIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    model_path = os.path.join(method_dir, "xgb_model.ubj")
    meta_path = os.path.join(method_dir, "meta.json")

    model.save_model(model_path)

    # meta.json must be JSON-serializable — coerce numpy types
    def _clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    clean_meta = {k: _clean(v) for k, v in meta.items()}
    with open(meta_path, "w") as f:
        json.dump(clean_meta, f, indent=2)

    if feat_imp_df is not None:
        feat_imp_path = os.path.join(method_dir, "feature_importance.csv")
        feat_imp_df.to_csv(feat_imp_path, index=False)

    print(f"[LOG] Model saved      -> {model_path}")
    print(f"[LOG] Metadata saved   -> {meta_path}")
    return method_dir


def load_model_artifacts(method_name):
    """Loads back a model saved by save_model_artifacts(). Useful for
    re-evaluating without retraining: see the commented helper at the
    bottom of each 0X_*.py script."""
    import xgboost as xgb

    method_dir = os.path.join(MODELS_DIR, method_name)
    model_path = os.path.join(method_dir, "xgb_model.ubj")
    meta_path = os.path.join(method_dir, "meta.json")

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"No saved model found at {method_dir}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta


def log_result(method_name, metrics, params=None, ratio_info=None, elapsed=None):
    """Append one method's results into the shared JSON log used by run_all.py
    to build the final comparison table. Safe to call from independently-run
    scripts since it reads-modifies-writes the same file."""
    entry = {
        "method": method_name,
        "metrics": metrics,
        "params": params or {},
        "ratio_info": ratio_info or {},
        "elapsed_sec": elapsed,
        "timestamp": time.time(),
    }

    log = []
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG) as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []

    # Replace any prior entry for the same method so re-running a script
    # updates its row instead of duplicating it.
    log = [e for e in log if e["method"] != method_name]
    log.append(entry)

    with open(RESULTS_LOG, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[LOG] Result for '{method_name}' saved -> {RESULTS_LOG}")
    return entry


def build_comparison_table():
    """Reads the shared log and returns a sorted pandas DataFrame."""
    if not os.path.exists(RESULTS_LOG):
        print("[WARN] No results logged yet.")
        return pd.DataFrame()

    with open(RESULTS_LOG) as f:
        log = json.load(f)

    rows = []
    for e in log:
        row = {"method": e["method"], "elapsed_sec": e.get("elapsed_sec")}
        row.update(e["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mcc", ascending=False).reset_index(drop=True)
    return df


def best_threshold_by_mcc(y_true, y_prob, n_steps=199):
    """Scans thresholds in (0, 1) and returns the one maximizing MCC.
    Used by 05_threshold_optimization.py but kept here since other
    scripts may want to report MCC-optimal threshold alongside default 0.5."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_mcc = 0.5, -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        m = matthews_corrcoef(y_true, preds)
        if m > best_mcc:
            best_mcc, best_t = m, t
    return best_t, best_mcc


def full_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)
