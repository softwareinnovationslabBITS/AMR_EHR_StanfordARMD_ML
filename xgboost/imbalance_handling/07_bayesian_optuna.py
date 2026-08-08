"""
07_bayesian_optuna.py
------------------------
Item 7: XGB + Bayesian optimization (via Optuna's TPE sampler).

FAST SETTINGS (per your preference): 30 trials, 3-fold CV inside the
objective function, with pruning so clearly-bad trials stop early instead
of running all 2000 boosting rounds.

Install if needed:
    pip install optuna optuna-integration[xgboost] --break-system-packages

Search space covers the parameters that matter most for XGBoost on tabular
data: max_depth, learning_rate, subsample, colsample_bytree, min_child_weight,
gamma, reg_alpha, reg_lambda, and scale_pos_weight (search around the
natural class ratio rather than assuming 1.0 is optimal).

Optimizes PR-AUC (average_precision) as the objective, since ROC-AUC can
look deceptively good under heavy class imbalance — PR-AUC is the more
honest target for a rare-positive-class problem like this one.

Saves the FINAL model (retrained on the full training set using the best
hyperparameters found by the search) + metadata (including the full best
params dict) to ./saved_models/07_bayesian_optuna/, and prints the full
classification report + top-15 feature importances.
"""

import time
import joblib
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from common import load_cached_split, PREPROCESSOR_PATH
from utils import (
    compute_metrics, print_report, log_result,
    print_classification_report, print_top_features, save_model_artifacts,
)

# XGBoostPruningCallback moved to the separate optuna_integration package in
# recent Optuna releases; fall back to the older optuna.integration path for
# older installs so this script runs either way.
try:
    from optuna_integration import XGBoostPruningCallback
except ImportError:
    from optuna.integration import XGBoostPruningCallback

# #migrate: Optuna search constants from the single config file
METHOD_NAME = "07_bayesian_optuna"
from common import RANDOM_STATE, OPTUNA_N_TRIALS, OPTUNA_N_SPLITS
N_TRIALS = OPTUNA_N_TRIALS
N_SPLITS = OPTUNA_N_SPLITS


# #migrate: use XGBoost device/tree_method from the single config file
def objective(trial, X_train, y_train, base_ratio):
    from common import EVAL_METRIC, TREE_METHOD, DEVICE
    params = {
        "objective": "binary:logistic",
        "eval_metric": EVAL_METRIC,
        "tree_method": TREE_METHOD,
        "device": DEVICE,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # Search scale_pos_weight around the natural ratio instead of
        # assuming the un-weighted natural ratio is optimal.
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", base_ratio * 0.5, base_ratio * 1.5),
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Pruning callback only attached to the first fold's eval, which is
        # enough signal to kill clearly-bad trials early without adding
        # complexity for multi-fold pruning aggregation.
        callbacks = []
        if fold_idx == 0:
            callbacks.append(XGBoostPruningCallback(trial, "validation_0-aucpr"))

        model = xgb.XGBClassifier(
            **params,
            n_estimators=1000,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            early_stopping_rounds=30,
            missing=np.nan,
            callbacks=callbacks if callbacks else None,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob_val = model.predict_proba(X_val)[:, 1]
        fold_scores.append(average_precision_score(y_val, y_prob_val))

    return float(np.mean(fold_scores))


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    print(f"[LOG] Loaded cached split — X_train {X_train.shape}, X_test {X_test.shape}")

    base_ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"[LOG] Base scale_pos_weight (natural ratio): {base_ratio:.3f}")
    print(f"[LOG] Running Optuna study: {N_TRIALS} trials, {N_SPLITS}-fold CV per trial, pruning enabled...")

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, base_ratio),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )
    search_elapsed = time.time() - t0

    print(f"\n[LOG] Optuna search finished in {search_elapsed:.1f}s")
    print(f"[LOG] Best CV PR-AUC: {study.best_value:.4f}")
    print(f"[LOG] Best params: {study.best_params}")
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"[LOG] {n_pruned}/{N_TRIALS} trials pruned early")

    # Retrain final model on FULL training set with best params, evaluate on
    # the held-out test set for a number comparable to every other script.
    # This is also the model that gets saved to disk.
    best_params = study.best_params
    print("\n[LOG] Retraining final model on full training set with best hyperparameters...")
    t0 = time.time()
    # #migrate: use XGBoost params from the single config file
    from common import N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH, SUBSAMPLE, COLSAMPLE_BYTREE, EVAL_METRIC, N_JOBS, EARLY_STOPPING_ROUNDS, TREE_METHOD, DEVICE
    # Note: the Optuna search overrides most hyperparameters, but we still
    # seed final_model with config defaults so unspecified params match config.
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=EVAL_METRIC,
        tree_method=TREE_METHOD,
        device=DEVICE,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        missing=np.nan,
        **best_params,
    )
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    final_elapsed = time.time() - t0

    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = final_model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    print_report(
        METHOD_NAME, metrics,
        extra={
            "n_trials": N_TRIALS,
            "n_pruned": n_pruned,
            "best_cv_pr_auc": round(study.best_value, 4),
            "best_iteration": final_model.best_iteration,
            **{f"best_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in best_params.items()},
        },
        elapsed=search_elapsed + final_elapsed,
    )
    print_classification_report(y_test, y_pred)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feat_imp_df = print_top_features(final_model, preprocessor, meta['cat_cols'], meta['num_cols'])

    run_meta = {
        "method": METHOD_NAME,
        "n_trials": N_TRIALS,
        "n_splits": N_SPLITS,
        "n_pruned": n_pruned,
        "best_cv_pr_auc": float(study.best_value),
        "best_params": best_params,
        "best_iteration": int(final_model.best_iteration),
        "metrics": metrics,
        "elapsed_sec": search_elapsed + final_elapsed,
        "cat_cols": meta['cat_cols'],
        "num_cols": meta['num_cols'],
    }
    save_model_artifacts(METHOD_NAME, final_model, run_meta, feat_imp_df)
    log_result(
        METHOD_NAME, metrics,
        params={"n_trials": N_TRIALS, "n_splits": N_SPLITS, **best_params},
        elapsed=search_elapsed + final_elapsed,
    )


# ==============================================================================
# CONVENIENCE: LOAD-ONLY HELPER
# ==============================================================================
# from utils import load_model_artifacts
# model, meta = load_model_artifacts("07_bayesian_optuna")
# print(meta["best_params"])

if __name__ == "__main__":
    main()
