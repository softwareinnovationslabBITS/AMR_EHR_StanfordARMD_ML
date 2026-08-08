"""
06_kfold_cv.py
---------------
Item 6: XGB + k-fold cross validation.

This script deliberately mirrors 01_baseline_xgb.py's hyperparameters
(no scale_pos_weight, no resampling) so the ONLY variable that changes is
"single train/test split" vs "3-fold stratified CV". That isolates what CV
alone contributes — if you stack CV with class weights or SMOTE, you can no
longer tell which part of the improvement came from which technique.

Uses StratifiedKFold so each fold preserves the overall S:R ratio. Reports
mean +/- std across folds, then ALSO retrains once on the full training set
and evaluates on the held-out test set, so this script's final number is
still comparable to every other script's "performance on X_test" row in the
comparison table. THIS final retrained model (not any individual fold's
model) is what gets saved to disk.

Fast settings per your preference: n_splits=3.

Saves model + metadata (including per-fold CV stats) to
./saved_models/06_kfold_cv/ and prints the full classification report +
top-15 feature importances for the final retrained model.
"""

import time
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from common import load_cached_split, PREPROCESSOR_PATH
from utils import (
    compute_metrics, print_report, log_result,
    print_classification_report, print_top_features, save_model_artifacts,
)

# #migrate: k-fold splits from the single config file
METHOD_NAME = "06_kfold_cv"
from common import RANDOM_STATE, KFOLD_N_SPLITS
N_SPLITS = KFOLD_N_SPLITS


# #migrate: use XGBoost params from the single config file
def make_model():
    from common import N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH, SUBSAMPLE, COLSAMPLE_BYTREE, EVAL_METRIC, N_JOBS, EARLY_STOPPING_ROUNDS, TREE_METHOD, DEVICE
    return xgb.XGBClassifier(
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


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    print(f"[LOG] Loaded cached split — X_train {X_train.shape}, X_test {X_test.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    t0 = time.time()
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = make_model()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob_val = model.predict_proba(X_val)[:, 1]
        y_pred_val = model.predict(X_val)
        m = compute_metrics(y_val, y_pred_val, y_prob_val)
        fold_metrics.append(m)
        print(f"[FOLD {fold_idx}/{N_SPLITS}] ROC-AUC={m['roc_auc']:.4f}  "
              f"PR-AUC={m['pr_auc']:.4f}  MCC={m['mcc']:.4f}")

    cv_elapsed = time.time() - t0

    # Aggregate CV stats across folds
    cv_summary = {}
    for key in fold_metrics[0].keys():
        vals = [m[key] for m in fold_metrics]
        cv_summary[f"{key}_mean"] = float(np.mean(vals))
        cv_summary[f"{key}_std"] = float(np.std(vals))

    print("\n[CV SUMMARY] (3-fold, mean +/- std)")
    for key in ['roc_auc', 'pr_auc', 'balanced_accuracy', 'mcc', 'cohen_kappa']:
        print(f"  {key:20s}: {cv_summary[key+'_mean']:.4f} +/- {cv_summary[key+'_std']:.4f}")

    # Final retrain on FULL training set, evaluated on the untouched test set,
    # so this row is comparable to every other script in the comparison table.
    # This is also the model that gets saved to disk.
    print("\n[LOG] Retraining on full training set for final test-set evaluation...")
    t0 = time.time()
    final_model = make_model()
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    final_elapsed = time.time() - t0

    y_prob_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = final_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test, y_prob_test)

    print_report(
        METHOD_NAME, test_metrics,
        extra={
            "cv_mcc_mean": round(cv_summary['mcc_mean'], 4),
            "cv_mcc_std": round(cv_summary['mcc_std'], 4),
            "n_splits": N_SPLITS,
            "best_iteration": final_model.best_iteration,
        },
        elapsed=cv_elapsed + final_elapsed,
    )
    print_classification_report(y_test, y_pred_test)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feat_imp_df = print_top_features(final_model, preprocessor, meta['cat_cols'], meta['num_cols'])

    run_meta = {
        "method": METHOD_NAME,
        "n_splits": N_SPLITS,
        "cv_summary": cv_summary,
        "best_iteration": int(final_model.best_iteration),
        "metrics": test_metrics,
        "elapsed_sec": cv_elapsed + final_elapsed,
        "cat_cols": meta['cat_cols'],
        "num_cols": meta['num_cols'],
    }
    save_model_artifacts(METHOD_NAME, final_model, run_meta, feat_imp_df)
    log_result(
        METHOD_NAME, test_metrics,
        params={"n_splits": N_SPLITS, "cv_summary": cv_summary},
        elapsed=cv_elapsed + final_elapsed,
    )


# ==============================================================================
# CONVENIENCE: LOAD-ONLY HELPER
# ==============================================================================
# from utils import load_model_artifacts
# model, meta = load_model_artifacts("06_kfold_cv")
# print(meta["cv_summary"])  # per-fold mean/std stats are saved here too

if __name__ == "__main__":
    main()
