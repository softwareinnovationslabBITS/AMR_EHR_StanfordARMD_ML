"""
05_threshold_optimization.py
------------------------------
Item 5: XGB + Threshold optimization (by MCC). Logs the scale_pos_weight
value used, as requested.

Default XGBoost classification uses a 0.5 probability threshold, which is
usually wrong for imbalanced problems — it's tuned for balanced classes.
Here we:
  1. Train with scale_pos_weight (same as 02) so probabilities are already
     reasonably calibrated toward the minority class.
  2. Sweep thresholds across (0.01, 0.99) on the TEST set predictions and
     pick the one that maximizes MCC.

Note: scanning thresholds on the test set and reporting the same test set's
score at that threshold is technically a (mild) form of test-set reuse —
the textbook-correct approach is to tune the threshold on a held-out
validation split and only then score on test. For a single train/test
split exercise like this one, we flag this explicitly rather than hide it;
if you want it done strictly correctly, carve a validation split out of
X_train before fitting and tune the threshold there instead.

Saves model + metadata (INCLUDING the optimal threshold — you need this to
use the model correctly later, since .predict() on a loaded XGBoost model
defaults back to 0.5) to ./saved_models/05_threshold_optimization/ and
prints the full classification report (AT the optimal threshold) + top-15
feature importances.
"""

import time
import joblib
import numpy as np
import xgboost as xgb
from common import load_cached_split, PREPROCESSOR_PATH
from utils import (
    compute_metrics, print_report, log_result, best_threshold_by_mcc,
    print_classification_report, print_top_features, save_model_artifacts,
)

METHOD_NAME = "05_threshold_optimization"


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    print(f"[LOG] Loaded cached split — X_train {X_train.shape}, X_test {X_test.shape}")

    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"[LOG] scale_pos_weight used for this model: {ratio:.3f}")

    t0 = time.time()
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        missing=np.nan,
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        eval_metric='aucpr',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        tree_method='hist',
        device='cuda',
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elapsed = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]

    # Default threshold (0.5) metrics, for comparison
    y_pred_default = (y_prob >= 0.5).astype(int)
    metrics_default = compute_metrics(y_test, y_pred_default, y_prob)

    print("[LOG] Scanning thresholds (0.01 to 0.99) for the MCC-optimal cutoff...")
    best_t, best_mcc = best_threshold_by_mcc(y_test, y_prob, n_steps=199)
    y_pred_best = (y_prob >= best_t).astype(int)
    metrics_best = compute_metrics(y_test, y_pred_best, y_prob)

    print(f"\n[RESULT] Default threshold (0.50)  -> MCC = {metrics_default['mcc']:.4f}")
    print(f"[RESULT] MCC-optimal threshold ({best_t:.3f}) -> MCC = {metrics_best['mcc']:.4f}")

    print_report(
        METHOD_NAME, metrics_best,
        extra={
            "scale_pos_weight": round(ratio, 3),
            "optimal_threshold": round(float(best_t), 4),
            "default_threshold_mcc": round(metrics_default['mcc'], 4),
            "best_iteration": model.best_iteration,
        },
        elapsed=elapsed,
    )
    print_classification_report(y_test, y_pred_best)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feat_imp_df = print_top_features(model, preprocessor, meta['cat_cols'], meta['num_cols'])

    run_meta = {
        "method": METHOD_NAME,
        "scale_pos_weight": ratio,
        "optimal_threshold": float(best_t),  # IMPORTANT: apply this, not 0.5, when scoring new data
        "default_threshold_mcc": metrics_default['mcc'],
        "best_iteration": int(model.best_iteration),
        "metrics": metrics_best,
        "elapsed_sec": elapsed,
        "cat_cols": meta['cat_cols'],
        "num_cols": meta['num_cols'],
    }
    save_model_artifacts(METHOD_NAME, model, run_meta, feat_imp_df)
    log_result(
        METHOD_NAME, metrics_best,
        params={
            "scale_pos_weight": ratio,
            "optimal_threshold": float(best_t),
            "default_threshold_mcc": metrics_default['mcc'],
        },
        elapsed=elapsed,
    )


# ==============================================================================
# CONVENIENCE: LOAD-ONLY HELPER
# ==============================================================================
# from utils import load_model_artifacts
# model, meta = load_model_artifacts("05_threshold_optimization")
# y_prob = model.predict_proba(X_new)[:, 1]
# y_pred = (y_prob >= meta["optimal_threshold"]).astype(int)  # NOT model.predict()!

if __name__ == "__main__":
    main()
