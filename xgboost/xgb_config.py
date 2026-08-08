"""Configuration for DL-feature-matched XGBoost workflow."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
_XGB_CFG = _CFG.get('xgboost', {})

# #migrate: bundle now lives in the shared dataset/ folder
DL_BUNDLE_PATH = resolve_path(_XGB_CFG.get('bundle_path', 'dataset/amr_analysis_bundle.joblib'))
OUT = Path('xgb_dl_matched_outputs_v1')
MODEL_DIR = OUT/'models'; PRED_DIR = OUT/'predictions'; RESULT_DIR = OUT/'results'
PLOT_DIR = OUT/'plots'; SHAP_DIR = OUT/'shap'; OPTUNA_DIR = OUT/'optuna'; LOG_DIR = OUT/'logs'
RANDOM_SEED = _CFG.get('seed', 42)
DEVICE = _XGB_CFG.get('device', 'cuda')
TREE_METHOD = _XGB_CFG.get('tree_method', 'hist')
N_JOBS = _XGB_CFG.get('n_jobs', -1)
BATCH_PREDICTION_ROWS = 200_000

BASE_XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='aucpr', n_estimators=2500,
    learning_rate=0.02, max_depth=8, min_child_weight=1, subsample=0.8,
    colsample_bytree=0.7, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
    early_stopping_rounds=75,
)

N_FOLDS = _XGB_CFG.get('n_folds', 5)
N_OPTUNA_TRIALS = _XGB_CFG.get('n_optuna_trials', 30)
OPTUNA_FOLDS = 5
OPTUNA_TIMEOUT_SECONDS = None
RUN_SMOTENC = True
SMOTENC_SAMPLING_STRATEGY = 0.50
SMOTENC_K_NEIGHBORS = 5
RUN_UNDERSAMPLING = True
UNDERSAMPLING_STRATEGY = 1.0
THRESHOLD_OBJECTIVE = _XGB_CFG.get('threshold_objective', 'mcc')
THRESHOLD_GRID_SIZE = 999
N_BOOTSTRAPS = _XGB_CFG.get('n_bootstraps', 2000)
BOOTSTRAP_CONFIDENCE = 0.95
SHAP_SAMPLE_SIZE = 5000
TOP_N_FEATURES = 30
TOP_N_DEPENDENCE = 5
MODEL_SEQUENCE = _XGB_CFG.get('strategies', ['baseline','class_weighted','smotenc','undersampling','threshold_optimized','fivefold_cv','optuna'])
