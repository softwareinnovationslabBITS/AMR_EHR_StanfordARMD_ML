"""
gen_viz_05.py
--------------
Generates the full visualization suite for the 05_threshold_optimization
model. Run 05_threshold_optimization.py first.

IMPORTANT: this model's saved meta.json includes "optimal_threshold" — the
MCC-optimal cutoff found during training, NOT 0.5. We pass that through as
the threshold_override so the confusion matrix / classification report in
metrics_report.txt reflect how this model is actually meant to be used,
not XGBoost's generic 0.5 default. The 03_confusion_matrices.png plot will
then show "optimal_threshold" vs "best-F1-on-this-run" side by side (these
may differ slightly since F1-optimal and MCC-optimal thresholds aren't
always identical).

Outputs everything to ./visualizations/05_threshold_optimization/
"""

import joblib
from common import load_cached_split, PREPROCESSOR_PATH
from utils import load_model_artifacts, get_feature_names
from viz_utils import generate_all_visualizations

METHOD_NAME = "05_threshold_optimization"
OUT_DIR = f"./visualizations/{METHOD_NAME}"


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    model, model_meta = load_model_artifacts(METHOD_NAME)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = get_feature_names(preprocessor, meta['cat_cols'], meta['num_cols'])

    optimal_threshold = model_meta.get("optimal_threshold", 0.5)
    print(f"[LOG] Using saved optimal_threshold = {optimal_threshold:.4f} as the reporting threshold "
          f"(NOT the generic 0.5 default).")

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
        threshold_override=optimal_threshold,
    )


if __name__ == "__main__":
    main()
