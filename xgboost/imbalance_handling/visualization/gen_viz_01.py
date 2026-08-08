# Source: /AMR_Stanford/py_codes/xg_classw/gen_viz_01.py
"""
gen_viz_01.py
--------------
Generates the full visualization suite for the 01_baseline model.
Run 01_baseline_xgb.py first so the model exists in ./saved_models/01_baseline/.

Outputs everything to ./visualizations/01_baseline/
"""

import joblib
from common import load_cached_split, PREPROCESSOR_PATH
from utils import load_model_artifacts, get_feature_names
from viz_utils import generate_all_visualizations

METHOD_NAME = "01_baseline"
OUT_DIR = f"./visualizations/{METHOD_NAME}"


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    model, model_meta = load_model_artifacts(METHOD_NAME)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = get_feature_names(preprocessor, meta['cat_cols'], meta['num_cols'])

    # NOTE: pass X_test directly (sparse) — generate_all_visualizations
    # handles sparse-to-dense conversion internally in a way that preserves
    # XGBoost's missing-value semantics. Do NOT call .toarray() here.
    generate_all_visualizations(
        method_name=METHOD_NAME,
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()

