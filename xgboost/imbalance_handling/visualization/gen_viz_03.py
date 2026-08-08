"""
gen_viz_03.py
--------------
Generates the full visualization suite for the 03_smote model.
Run 03_smote.py first.

IMPORTANT: the 03_smote model was trained on MEDIAN-IMPUTED data (SMOTE's
k-NN can't handle NaN). We must re-apply the same imputation to X_test
before evaluating, or the model will see a different feature distribution
than it was trained on. The imputer is refit here on X_train for
consistency with 03_smote.py's own logic (SimpleImputer has no persisted
state to load, so we reproduce the exact same fit-on-train step).

CRITICAL: SimpleImputer.transform() on sparse input returns SPARSE output
(scipy keeps it sparse), and crucially writes the median (often 0 for
mostly-zero columns) into formerly-NaN positions as EXPLICIT stored
entries — while the data's original, always-real zeros remain UNSTORED
(implicit) entries, exactly as before imputation. This creates two
genuinely different kinds of zero that only sparse representation can
tell apart, and a naive .toarray() call here collapses that distinction
and produces badly wrong predictions (confirmed: ROC-AUC dropped from the
model's true 0.8768 down to 0.6271 this way). We pass the imputed matrix
through to generate_all_visualizations() STILL SPARSE — it converts to
dense internally using sparse_to_dense_preserving_missing(), which
reconstructs exactly what the model saw during its own training-time
evaluation. Do NOT call .toarray() on X_test_imp anywhere in this script.

Outputs everything to ./visualizations/03_smote/
"""

import joblib
from sklearn.impute import SimpleImputer
from common import load_cached_split, PREPROCESSOR_PATH
from utils import load_model_artifacts, get_feature_names
from viz_utils import generate_all_visualizations

METHOD_NAME = "03_smote"
OUT_DIR = f"./visualizations/{METHOD_NAME}"


def main():
    X_train, X_test, y_train, y_test, meta = load_cached_split()
    model, model_meta = load_model_artifacts(METHOD_NAME)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = get_feature_names(preprocessor, meta['cat_cols'], meta['num_cols'])

    print("[LOG] Re-applying median imputation (fit on train) to match how this model was trained...")
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)
    X_test_imp = imputer.transform(X_test)  # stays sparse — DO NOT densify here

    generate_all_visualizations(
        method_name=METHOD_NAME,
        model=model,
        X_test=X_test_imp,
        y_test=y_test,
        feature_names=feature_names,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()

