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
import json
import numpy as np
import scipy.sparse as sp

CACHE_DIR = "./cache"

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
            "No cached split found in ./cache/. Run `python 00_common.py` first."
        )
    X_train = sp.load_npz(X_TRAIN_PATH)
    X_test = sp.load_npz(X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return X_train, X_test, y_train, y_test, meta
