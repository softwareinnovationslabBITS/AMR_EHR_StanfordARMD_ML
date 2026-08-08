"""
common.py
---------
Importable companion to 00_common.py.

Python module names can't start with a digit, so `00_common.py` (the script
you actually RUN to build the cache) can't be `import`-ed by the other
scripts. This tiny module holds the shared path constants and the
`load_cached_split()` helper so 01-07 can do:

    from common import load_cached_split
    X_train, X_test, y_train, y_test, meta = load_cached_split()

without caring about the numeric filename.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

# #migrate: load paths and seed from the single config file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
_XGB_IMB_CFG = _CFG.get('xgboost_imbalance', {})
SEED = _CFG.get('seed', 42)

CACHE_DIR = str(resolve_path(_XGB_IMB_CFG.get('cache_dir', 'xgboost/imbalance_handling/cache')))
TEST_SIZE = _XGB_IMB_CFG.get('test_size', 0.2)
RANDOM_STATE = _XGB_IMB_CFG.get('random_state', SEED)
DEVICE = _XGB_IMB_CFG.get('device', 'cuda')
TREE_METHOD = _XGB_IMB_CFG.get('tree_method', 'hist')
N_JOBS = _XGB_IMB_CFG.get('n_jobs', -1)
N_ESTIMATORS = _XGB_IMB_CFG.get('n_estimators', 2000)
LEARNING_RATE = _XGB_IMB_CFG.get('learning_rate', 0.02)
MAX_DEPTH = _XGB_IMB_CFG.get('max_depth', 8)
SUBSAMPLE = _XGB_IMB_CFG.get('subsample', 0.8)
COLSAMPLE_BYTREE = _XGB_IMB_CFG.get('colsample_bytree', 0.7)
EVAL_METRIC = _XGB_IMB_CFG.get('eval_metric', 'aucpr')
EARLY_STOPPING_ROUNDS = _XGB_IMB_CFG.get('early_stopping_rounds', 50)
SMOTE_K_NEIGHBORS = _XGB_IMB_CFG.get('smote_k_neighbors', 5)
KFOLD_N_SPLITS = _XGB_IMB_CFG.get('kfold_n_splits', 3)
OPTUNA_N_TRIALS = _XGB_IMB_CFG.get('optuna_n_trials', 30)
OPTUNA_N_SPLITS = _XGB_IMB_CFG.get('optuna_n_splits', 3)

X_TRAIN_PATH = os.path.join(CACHE_DIR, "X_train.npz")
X_TEST_PATH = os.path.join(CACHE_DIR, "X_test.npz")
Y_TRAIN_PATH = os.path.join(CACHE_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(CACHE_DIR, "y_test.npy")
PREPROCESSOR_PATH = os.path.join(CACHE_DIR, "preprocessor.joblib")
META_PATH = os.path.join(CACHE_DIR, "meta.json")


def load_cached_split():
    """Loads the preprocessed train/test split built by 00_common.py.

    Returns
    -------
    X_train : scipy.sparse.csr_matrix
    X_test  : scipy.sparse.csr_matrix
    y_train : np.ndarray
    y_test  : np.ndarray
    meta    : dict (cat_cols, num_cols, test_size, random_state, n_features)
    """
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"No cached split found in {CACHE_DIR}/. Run `python 00_preprocess.py` first."
        )
    X_train = sp.load_npz(X_TRAIN_PATH)
    X_test = sp.load_npz(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return X_train, X_test, y_train, y_test, meta
