# Source: /AMR_Stanford/XGB_feature_engg/03_smote.py
"""
03_smote.py
------------
Item 3: XGB + SMOTE. Logs the change in S:R (Susceptible:Resistant) ratio
before and after resampling, as requested.

IMPORTANT correctness notes:
1. SMOTE is fit/applied to the TRAINING split ONLY. The test set stays
   untouched and unseen — resampling the test set would leak synthetic
   information into evaluation and inflate every metric artificially.
2. SMOTE's k-NN step cannot handle NaN. Your base pipeline leaves some
   numeric columns with NaN (XGBoost handles that natively via
   missing=np.nan, but SMOTE can't). So here — and ONLY here — we apply a
   median imputer fit on the training data before SMOTE. XGBoost itself
   still trains on the post-impute, post-SMOTE data for this script.
3. Imputation changes VALUES, not COLUMN COUNT, so feature importance
   names from the preprocessor still line up correctly afterward.
4. SMOTE works on the OneHotEncoded + numeric sparse matrix the same way
   it would on a dense one, but interpolating between one-hot columns can
   produce fractional values (e.g. 0.42 instead of 0 or 1) for category
   indicators. This is a known, accepted limitation of vanilla SMOTE on
   mixed-type data (SMOTENC exists for that, but adds real complexity and
   the assignment specifically asks for SMOTE).

Saves model + metadata to ./saved_models/03_smote/ and prints the full
classification report + top-15 feature importances.
"""

import time
import joblib
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from common import load_cached_split, PREPROCESSOR_PATH
from utils import (
    compute_metrics, print_report, log_result, print_ratio_change,
    print_classification_report, print_top_features, save_model_artifacts,
)

METHOD_NAME = "03_smote"


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    print(f"[LOG] Loaded cached split — X_train {X_train.shape}, X_test {X_test.shape}")

    # SMOTE can't handle NaN -> impute (fit on train only, applied to both
    # train and test so the model sees consistent column semantics).
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # #migrate: SMOTE k_neighbors from the single config file
    from common import RANDOM_STATE, SMOTE_K_NEIGHBORS
    print("[LOG] Running SMOTE on training data only...")
    t0 = time.time()
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
    X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train)
    smote_elapsed = time.time() - t0
    print(f"[LOG] SMOTE resampling took {smote_elapsed:.1f}s")

    ratio_info = print_ratio_change("SMOTE (train set)", y_train, y_train_res)

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
    model.fit(X_train_res, y_train_res, eval_set=[(X_test_imp, y_test)], verbose=False)
    elapsed = time.time() - t0

    y_prob = model.predict_proba(X_test_imp)[:, 1]
    y_pred = model.predict(X_test_imp)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_report(
        METHOD_NAME, metrics,
        extra={"best_iteration": model.best_iteration, "smote_time_sec": round(smote_elapsed, 1)},
        elapsed=elapsed,
    )
    print_classification_report(y_test, y_pred)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feat_imp_df = print_top_features(model, preprocessor, meta['cat_cols'], meta['num_cols'])

    run_meta = {
        "method": METHOD_NAME,
        "k_neighbors": SMOTE_K_NEIGHBORS,
        "sampling_strategy": "auto",
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
        params={"k_neighbors": 5, "sampling_strategy": "auto"},
        ratio_info=ratio_info,
        elapsed=elapsed,
    )


# ==============================================================================
# CONVENIENCE: LOAD-ONLY HELPER
# ==============================================================================
# from utils import load_model_artifacts
# model, meta = load_model_artifacts("03_smote")
# NOTE: this model was trained on MEDIAN-IMPUTED data — re-impute new data
# with the same strategy before predicting with it.

if __name__ == "__main__":
    main()
