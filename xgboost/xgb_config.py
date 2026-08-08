"""Configuration for DL-feature-matched XGBoost workflow."""
from pathlib import Path

# #migrate: bundle now lives in the shared dataset/ folder
DL_BUNDLE_PATH = Path('../dataset/amr_analysis_bundle.joblib')
OUT = Path('xgb_dl_matched_outputs_v1')
MODEL_DIR = OUT/'models'; PRED_DIR = OUT/'predictions'; RESULT_DIR = OUT/'results'
PLOT_DIR = OUT/'plots'; SHAP_DIR = OUT/'shap'; OPTUNA_DIR = OUT/'optuna'; LOG_DIR = OUT/'logs'
RANDOM_SEED = 42
DEVICE = 'cuda'
TREE_METHOD = 'hist'
N_JOBS = -1
BATCH_PREDICTION_ROWS = 200_000

BASE_XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='aucpr', n_estimators=2500,
    learning_rate=0.02, max_depth=8, min_child_weight=1, subsample=0.8,
    colsample_bytree=0.7, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
    early_stopping_rounds=75,
)

N_FOLDS = 5
N_OPTUNA_TRIALS = 30
OPTUNA_FOLDS = 5
OPTUNA_TIMEOUT_SECONDS = None
RUN_SMOTENC = True
SMOTENC_SAMPLING_STRATEGY = 0.50
SMOTENC_K_NEIGHBORS = 5
RUN_UNDERSAMPLING = True
UNDERSAMPLING_STRATEGY = 1.0
THRESHOLD_OBJECTIVE = 'mcc'
THRESHOLD_GRID_SIZE = 999
N_BOOTSTRAPS = 2000
BOOTSTRAP_CONFIDENCE = 0.95
SHAP_SAMPLE_SIZE = 5000
TOP_N_FEATURES = 30
TOP_N_DEPENDENCE = 5
MODEL_SEQUENCE = ['baseline','class_weighted','smotenc','undersampling','threshold_optimized','fivefold_cv','optuna']
