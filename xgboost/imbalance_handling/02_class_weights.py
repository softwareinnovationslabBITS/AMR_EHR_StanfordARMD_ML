"""
02_class_weights.py
---------------------
Item 2: XGB + Class weights, using the `scale_pos_weight` parameter.

scale_pos_weight = (# negative / # positive) in the TRAINING set only.
This penalizes mistakes on the minority (Resistant) class more heavily
during training, without changing the actual data distribution.

Saves model + metadata to ./saved_models/02_class_weights/ and prints the
full classification report + top-15 feature importances.
"""

import time
import joblib
import numpy as np
import xgboost as xgb
from common import load_cached_split, PREPROCESSOR_PATH
from utils import (
    compute_metrics, print_report, log_result,
    print_classification_report, print_top_features, save_model_artifacts,
)

METHOD_NAME = "02_class_weights"


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    print(f"[LOG] Loaded cached split — X_train {X_train.shape}, X_test {X_test.shape}")

    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"[LOG] Computed scale_pos_weight = {ratio:.3f}")

    # #migrate: use XGBoost params from the single config file
    t0 = time.time()
    from common import N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH, SUBSAMPLE, COLSAMPLE_BYTREE, EVAL_METRIC, N_JOBS, RANDOM_STATE, EARLY_STOPPING_ROUNDS, TREE_METHOD, DEVICE
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        missing=np.nan,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        eval_metric=EVAL_METRIC,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        tree_method=TREE_METHOD,
        device=DEVICE,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elapsed = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_report(
        METHOD_NAME, metrics,
        extra={"scale_pos_weight": round(ratio, 3), "best_iteration": model.best_iteration},
        elapsed=elapsed,
    )
    print_classification_report(y_test, y_pred)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feat_imp_df = print_top_features(model, preprocessor, meta['cat_cols'], meta['num_cols'])

    run_meta = {
        "method": METHOD_NAME,
        "scale_pos_weight": ratio,
        "best_iteration": int(model.best_iteration),
        "metrics": metrics,
        "elapsed_sec": elapsed,
        "cat_cols": meta['cat_cols'],
        "num_cols": meta['num_cols'],
    }
    save_model_artifacts(METHOD_NAME, model, run_meta, feat_imp_df)
    log_result(METHOD_NAME, metrics, params={"scale_pos_weight": ratio}, elapsed=elapsed)


# ==============================================================================
# CONVENIENCE: LOAD-ONLY HELPER
# ==============================================================================
# from utils import load_model_artifacts
# model, meta = load_model_artifacts("02_class_weights")

if __name__ == "__main__":
    main()
