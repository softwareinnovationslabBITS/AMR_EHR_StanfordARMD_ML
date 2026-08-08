"""
04_undersampling.py
---------------------
Item 4: XGB + Under sampling. Logs the change in S:R ratio before and after.

RandomUnderSampler just drops rows at random from the majority class — it
doesn't compute distances, so unlike SMOTE it works fine directly on the
sparse, NaN-containing matrix. No imputation needed here.

Applied to the TRAINING split only; test set is left exactly as-is.

Saves model + metadata to ./saved_models/04_undersampling/ and prints the
full classification report + top-15 feature importances.
"""

import time
import joblib
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
from common import load_cached_split, PREPROCESSOR_PATH
from utils import (
    compute_metrics, print_report, log_result, print_ratio_change,
    print_classification_report, print_top_features, save_model_artifacts,
)

METHOD_NAME = "04_undersampling"


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    print(f"[LOG] Loaded cached split — X_train {X_train.shape}, X_test {X_test.shape}")

    # #migrate: random_state from the single config file
    from common import RANDOM_STATE
    print("[LOG] Running Random Undersampling on training data only...")
    t0 = time.time()
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    rus_elapsed = time.time() - t0
    print(f"[LOG] Undersampling took {rus_elapsed:.2f}s")

    ratio_info = print_ratio_change("Undersampling (train set)", y_train, y_train_res)
    print(f"[LOG] Training rows: {X_train.shape[0]:,} -> {X_train_res.shape[0]:,} "
          f"(majority class cut down to match minority count)")

    # #migrate: use XGBoost params from the single config file
    t0 = time.time()
    from common import N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH, SUBSAMPLE, COLSAMPLE_BYTREE, EVAL_METRIC, N_JOBS, EARLY_STOPPING_ROUNDS, TREE_METHOD, DEVICE
    model = xgb.XGBClassifier(
        objective='binary:logistic',
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
    model.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=False)
    elapsed = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_report(
        METHOD_NAME, metrics,
        extra={"best_iteration": model.best_iteration, "undersample_time_sec": round(rus_elapsed, 2)},
        elapsed=elapsed,
    )
    print_classification_report(y_test, y_pred)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feat_imp_df = print_top_features(model, preprocessor, meta['cat_cols'], meta['num_cols'])

    run_meta = {
        "method": METHOD_NAME,
        "sampling_strategy": "auto (1:1)",
        "ratio_info": ratio_info,
        "best_iteration": int(model.best_iteration),
        "metrics": metrics,
        "elapsed_sec": elapsed,
        "cat_cols": meta['cat_cols'],
        "num_cols": meta['num_cols'],
    }
    save_model_artifacts(METHOD_NAME, model, run_meta, feat_imp_df)
    log_result(
        METHOD_NAME, metrics,
        params={"sampling_strategy": "auto (1:1)"},
        ratio_info=ratio_info,
        elapsed=elapsed,
    )


# ==============================================================================
# CONVENIENCE: LOAD-ONLY HELPER
# ==============================================================================
# from utils import load_model_artifacts
# model, meta = load_model_artifacts("04_undersampling")

if __name__ == "__main__":
    main()
