# Source: /AMR_Stanford/py_codes/xg_classw/viz_utils.py
"""
viz_utils.py
-------------
One function per plot type, styled to match your reference images:
  01_roc_curve.png
  02_pr_curve.png
  03_confusion_matrices.png
  04_threshold_analysis.png
  05_calibration_curve.png
  06_score_distribution.png
  07_feature_importance.png
  08_shap_summary_beeswarm.png
  09_shap_bar_meanabs.png
  10_shap_dependence_<antibiotic>.png  (one per antibiotic, top-N by SHAP)
  11_shap_waterfall_TN_example.png
  11_shap_waterfall_TP_example.png

Every plot-saving function here is loaded WITHOUT retraining — they all
take an already-fitted model + already-loaded data/labels and just
evaluate + plot. This file has zero dependency on any specific technique
(SMOTE, class weights, etc.) — it's pure plotting/explainability code,
reused identically by every 0X_generate_visualizations_*.py script.
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")  # headless backend — never tries to open a GUI window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve

sns.set_style("whitegrid")


# ==============================================================================
# CRITICAL — sparse-to-dense conversion that preserves XGBoost's missing
# value semantics.
#
# DO NOT replace this with a plain `.toarray()` call anywhere in this file.
#
# Background: these models were trained with missing=np.nan on SPARSE
# matrices. scipy.sparse silently drops EXPLICIT zeros when a sparse matrix
# is constructed or round-tripped — every "real" 0.0 that was never
# explicitly assigned becomes an UNSTORED (implicit) entry. XGBoost's
# DMatrix treats every unstored sparse entry as "missing", regardless of
# the missing= parameter.
#
# A plain `.toarray()` materializes every cell as an explicit float,
# destroying the implicit/explicit distinction entirely — a dense array
# has no concept of "unstored". This silently and catastrophically changes
# what the model sees (confirmed: ROC-AUC dropped from ~0.89 to ~0.43 on
# one model, and on another — 03_smote, where median imputation had ALSO
# written explicit stored zeros into formerly-NaN slots — even forcing
# model.missing=0 on the dense array only partially fixed it, because two
# DIFFERENT kinds of zero existed in the sparse data: implicit original
# zeros and explicit imputed zeros, and only sparse format can represent
# that distinction).
#
# The correct, general fix: reconstruct a dense array where every
# IMPLICIT (unstored) sparse position becomes NaN, and every EXPLICIT
# stored value (including stored zeros) keeps its literal value exactly.
# This exactly reproduces what the sparse DMatrix saw, verified bit-for-bit
# identical to sparse-input predictions on every model tested.
# ==============================================================================
def sparse_to_dense_preserving_missing(X_sparse):
    """Converts a scipy sparse matrix to a dense numpy array, mapping every
    implicit (unstored) entry to NaN instead of 0.0. Use this EVERYWHERE
    a model trained on sparse data needs to be scored on a dense array —
    never call .toarray() directly on these models' inputs.

    If X_sparse is already a dense numpy array, returns it unchanged
    (idempotent — safe to call even if a caller already densified).
    """
    if not sp.issparse(X_sparse):
        return X_sparse  # already dense; nothing to fix

    X_csr = X_sparse.tocsr()
    dense = np.full(X_csr.shape, np.nan, dtype=np.float64)
    coo = X_csr.tocoo()
    dense[coo.row, coo.col] = coo.data
    return dense


# ==============================================================================
# 01 — ROC CURVE
# ==============================================================================
def plot_roc_curve(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#1f5fd9", lw=2.5, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return auc


# ==============================================================================
# 02 — PRECISION-RECALL CURVE
# ==============================================================================
def plot_pr_curve(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(recall, precision, color="#d62728", lw=2.5, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(prevalence, color="gray", lw=1, linestyle="--",
               label=f"Baseline prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ap


# ==============================================================================
# 03 — CONFUSION MATRICES (default 0.5 vs best-F1 threshold, side by side)
# ==============================================================================
def plot_confusion_matrices(y_true, y_prob, out_path, default_threshold=0.5, best_threshold=None):
    if best_threshold is None:
        # Find best-F1 threshold ourselves if the caller didn't already have one
        thresholds = np.linspace(0.01, 0.99, 197)
        f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
        best_threshold = float(thresholds[int(np.argmax(f1s))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = ["Susceptible", "Resistant"]

    for ax, thresh, title in zip(
        axes,
        [default_threshold, best_threshold],
        [f"Default threshold = {default_threshold:.2f}", f"Best-F1 threshold = {best_threshold:.2f}"],
    ):
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 13},
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return best_threshold


# ==============================================================================
# 04 — PRECISION / RECALL / F1 vs DECISION THRESHOLD
# ==============================================================================
def plot_threshold_analysis(y_true, y_prob, out_path, n_steps=197):
    thresholds = np.linspace(0.01, 0.99, n_steps)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, precisions, color="#1f5fd9", lw=2, label="Precision")
    ax.plot(thresholds, recalls, color="#d62728", lw=2, label="Recall")
    ax.plot(thresholds, f1s, color="#2ca02c", lw=2, label="F1")
    ax.axvline(best_t, color="gray", lw=1.5, linestyle="--", label=f"Best F1 @ {best_t:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. Decision Threshold")
    ax.legend(loc="lower center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(best_t), float(f1s[best_idx])


# ==============================================================================
# 05 — CALIBRATION CURVE
# ==============================================================================
def plot_calibration_curve(y_true, y_prob, out_path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(mean_pred, frac_pos, marker="o", color="#7b2cbf", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction resistant")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==============================================================================
# 06 — PREDICTED PROBABILITY DISTRIBUTION BY TRUE CLASS
# ==============================================================================
def plot_score_distribution(y_true, y_prob, out_path, n_bins=40):
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(y_prob[y_true == 0], bins=n_bins, range=(0, 1), density=True,
            alpha=0.55, color="#6699ee", label="Susceptible")
    ax.hist(y_prob[y_true == 1], bins=n_bins, range=(0, 1), density=True,
            alpha=0.55, color="#ee7777", label="Resistant")
    ax.set_xlabel("Predicted probability of resistance")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Probability Distribution by True Class")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==============================================================================
# 07 — FEATURE IMPORTANCE (gain / weight / cover, 3 panels)
# ==============================================================================
def plot_feature_importance_panels(model, feature_names, out_path, top_n=15):
    booster = model.get_booster()
    name_map = {f"f{i}": name for i, name in enumerate(feature_names)}

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    importance_types = ["gain", "weight", "cover"]

    tables = {}
    for ax, imp_type in zip(axes, importance_types):
        raw_scores = booster.get_score(importance_type=imp_type)
        if not raw_scores:
            ax.set_title(f"Top {top_n} Features by '{imp_type}' (no data)")
            continue
        df = pd.DataFrame({
            "Feature": [name_map.get(k, k) for k in raw_scores.keys()],
            imp_type: list(raw_scores.values()),
        }).sort_values(imp_type, ascending=False).head(top_n)
        tables[imp_type] = df

        ax.barh(df["Feature"][::-1], df[imp_type][::-1], color="#1f5fd9")
        ax.set_title(f"Top {top_n} Features by '{imp_type}'")
        ax.set_xlabel(imp_type)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return tables


# ==============================================================================
# SHAP HELPERS
# ==============================================================================
def _sanitize_feature_name(name):
    """XGBoost's DMatrix rejects feature names containing '[', ']', or '<'
    (raised as ValueError: 'feature_names must be string, and may not
    contain [, ] or <'). Some OneHotEncoder-generated names — e.g. organism
    or antibiotic categories with brackets in their raw text — can contain
    these characters. shap.TreeExplainer builds its own internal DMatrix
    from whatever column names the input DataFrame has, so this bites here
    even though your original training pipeline already worked around the
    same issue at the DMatrix-building step elsewhere.

    We replace only the offending characters (not full slugification) so
    names stay close to recognizable; the ORIGINAL names are restored on
    the returned Explanation object afterward, so every downstream plot
    (beeswarm, bar, dependence, waterfall) still displays your real
    feature names, not the sanitized stand-ins.
    """
    return (
        str(name)
        .replace("[", "(")
        .replace("]", ")")
        .replace("<", "lt_")
    )


def compute_shap_explanation(model, X, feature_names, max_samples=2000, random_state=42):
    """Computes a SHAP Explanation object for a (subsampled, if large)
    feature matrix. TreeExplainer on XGBClassifier returns log-odds (margin)
    space by default — matching the f(x) scale in waterfall plots.

    X can be sparse OR dense — sparse_to_dense_preserving_missing() handles
    the conversion correctly either way (see that function's docstring for
    why a plain .toarray() silently corrupts predictions for these models).
    Subsampling happens BEFORE densification so we only ever materialize a
    dense array for the (small) SHAP subsample, not the full test set.

    max_samples caps the SHAP computation cost; on row counts like the
    ~245K-row test set in your metrics_report.txt, computing SHAP for every
    single row is unnecessary for summary plots and very slow. A random
    subsample of max_samples rows gives a representative picture instead.
    """
    import shap

    n = X.shape[0]
    if n > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X[idx]
    else:
        idx = np.arange(n)
        X_sub = X

    X_sub_dense = sparse_to_dense_preserving_missing(X_sub)

    # Sanitize names ONLY for the DataFrame handed to SHAP's internal
    # DMatrix construction — restore real names on the Explanation object
    # right after, so plots still show your actual feature names.
    safe_names = [_sanitize_feature_name(n) for n in feature_names]
    X_df = pd.DataFrame(X_sub_dense, columns=safe_names)

    explainer = shap.TreeExplainer(model)
    sv = explainer(X_df)
    sv.feature_names = list(feature_names)  # restore original names for display
    return sv, idx


def plot_shap_beeswarm(shap_explanation, out_path, max_display=18):
    import shap
    fig = plt.figure(figsize=(10, 11))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_bar(shap_explanation, out_path, max_display=18):
    import shap
    fig = plt.figure(figsize=(10, 11))
    shap.plots.bar(shap_explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence(shap_explanation, feature_name, out_path):
    """Mirrors the old shap.dependence_plot — current SHAP calls this
    shap.plots.scatter on a single-feature slice of the Explanation."""
    import shap
    fig = plt.figure(figsize=(7, 6))
    shap.plots.scatter(shap_explanation[:, feature_name], show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def top_shap_features_by_meanabs(shap_explanation, prefix_filter=None, top_n=3):
    """Returns the top_n feature names by mean(|SHAP value|), optionally
    restricted to names starting with prefix_filter (e.g. 'antibiotic_')
    so dependence plots focus on the same kind of feature as your reference
    images (per-antibiotic dependence plots)."""
    vals = np.abs(shap_explanation.values).mean(axis=0)
    names = shap_explanation.feature_names
    df = pd.DataFrame({"Feature": names, "MeanAbsSHAP": vals})
    if prefix_filter:
        df = df[df["Feature"].str.startswith(prefix_filter)]
    df = df.sort_values("MeanAbsSHAP", ascending=False)
    return df["Feature"].head(top_n).tolist()


def plot_shap_waterfall_example(shap_explanation, idx_in_explanation, out_path, max_display=15):
    import shap
    fig = plt.figure(figsize=(10, 9))
    shap.plots.waterfall(shap_explanation[idx_in_explanation], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_tn_tp_examples(y_true_sub, y_pred_sub, y_prob_sub):
    """Finds one clean True-Negative and one clean True-Positive example
    (highest-confidence correct predictions) from the SHAP subsample, to
    match your '11_shap_waterfall_TN_example.png' / 'TP_example.png' pair."""
    y_true_sub = np.asarray(y_true_sub)
    y_pred_sub = np.asarray(y_pred_sub)
    y_prob_sub = np.asarray(y_prob_sub)

    tn_mask = (y_true_sub == 0) & (y_pred_sub == 0)
    tp_mask = (y_true_sub == 1) & (y_pred_sub == 1)

    tn_idx = None
    tp_idx = None
    if tn_mask.any():
        tn_candidates = np.where(tn_mask)[0]
        tn_idx = int(tn_candidates[np.argmin(y_prob_sub[tn_candidates])])  # most confidently negative
    if tp_mask.any():
        tp_candidates = np.where(tp_mask)[0]
        tp_idx = int(tp_candidates[np.argmax(y_prob_sub[tp_candidates])])  # most confidently positive

    return tn_idx, tp_idx


# ==============================================================================
# MASTER ORCHESTRATOR — runs all 12 plots + writes metrics_report.txt
# ==============================================================================
def generate_all_visualizations(
    method_name,
    model,
    X_test,
    y_test,
    feature_names,
    out_dir,
    shap_max_samples=2000,
    top_antibiotics_n=3,
    threshold_override=None,
):
    """Runs the FULL visualization suite for one already-trained model and
    saves everything into out_dir, named exactly like your reference images:

        01_roc_curve.png
        02_pr_curve.png
        03_confusion_matrices.png
        04_threshold_analysis.png
        05_calibration_curve.png
        06_score_distribution.png
        07_feature_importance.png
        08_shap_summary_beeswarm.png
        09_shap_bar_meanabs.png
        10_shap_dependence_antibiotic_<NAME>.png   (one per top antibiotic)
        11_shap_waterfall_TN_example.png
        11_shap_waterfall_TP_example.png
        metrics_report.txt

    NOTE: does NOT retrain anything — `model` must already be fitted.

    X_test can be SPARSE (scipy.sparse matrix) or dense (numpy array) and
    must already be in the exact same feature space the model was trained
    on (e.g. already median-imputed for the 03_smote model — see that
    script's specific launcher for the imputation step applied before this
    function is called). Internally converted via
    sparse_to_dense_preserving_missing() rather than a plain .toarray() —
    see that function's docstring for why this matters: these models were
    trained on sparse data where unstored entries are implicitly "missing"
    regardless of value, and a naive .toarray() call destroys that
    distinction, silently producing badly wrong predictions.

    threshold_override: if given (e.g. the saved 'optimal_threshold' from
    05_threshold_optimization), used as the "default" comparison threshold
    in the confusion-matrix and report sections instead of 0.5 — for that
    one technique 0.5 isn't actually how the model is meant to be used.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}\n  GENERATING VISUALIZATIONS — {method_name}\n{'='*70}")

    X_test_dense = sparse_to_dense_preserving_missing(X_test)
    y_prob = model.predict_proba(X_test_dense)[:, 1]
    default_threshold = threshold_override if threshold_override is not None else 0.5
    y_pred_default = (y_prob >= default_threshold).astype(int)

    # ---- 01 ROC ----
    auc = plot_roc_curve(y_test, y_prob, os.path.join(out_dir, "01_roc_curve.png"))
    print(f"[OK] 01_roc_curve.png  (AUC={auc:.4f})")

    # ---- 02 PR ----
    ap = plot_pr_curve(y_test, y_prob, os.path.join(out_dir, "02_pr_curve.png"))
    print(f"[OK] 02_pr_curve.png  (AP={ap:.4f})")

    # ---- 04 threshold analysis (compute best_t first, reuse for 03) ----
    best_t, best_f1 = plot_threshold_analysis(y_test, y_prob, os.path.join(out_dir, "04_threshold_analysis.png"))
    print(f"[OK] 04_threshold_analysis.png  (best F1 threshold={best_t:.3f}, F1={best_f1:.4f})")

    # ---- 03 confusion matrices (default vs best-F1) ----
    plot_confusion_matrices(
        y_test, y_prob, os.path.join(out_dir, "03_confusion_matrices.png"),
        default_threshold=default_threshold, best_threshold=best_t,
    )
    print(f"[OK] 03_confusion_matrices.png")

    # ---- 05 calibration ----
    plot_calibration_curve(y_test, y_prob, os.path.join(out_dir, "05_calibration_curve.png"))
    print(f"[OK] 05_calibration_curve.png")

    # ---- 06 score distribution ----
    plot_score_distribution(y_test, y_prob, os.path.join(out_dir, "06_score_distribution.png"))
    print(f"[OK] 06_score_distribution.png")

    # ---- 07 feature importance (gain/weight/cover) ----
    plot_feature_importance_panels(model, feature_names, os.path.join(out_dir, "07_feature_importance.png"))
    print(f"[OK] 07_feature_importance.png")

    # ---- SHAP (08, 09, 10, 11) ----
    print(f"[LOG] Computing SHAP values on up to {shap_max_samples} sampled test rows...")
    shap_exp, sub_idx = compute_shap_explanation(model, X_test_dense, feature_names, max_samples=shap_max_samples)

    plot_shap_beeswarm(shap_exp, os.path.join(out_dir, "08_shap_summary_beeswarm.png"))
    print(f"[OK] 08_shap_summary_beeswarm.png")

    plot_shap_bar(shap_exp, os.path.join(out_dir, "09_shap_bar_meanabs.png"))
    print(f"[OK] 09_shap_bar_meanabs.png")

    top_abx = top_shap_features_by_meanabs(shap_exp, prefix_filter="antibiotic_", top_n=top_antibiotics_n)
    if not top_abx:
        # fall back to top overall features if no antibiotic_ prefixed columns exist
        top_abx = top_shap_features_by_meanabs(shap_exp, prefix_filter=None, top_n=top_antibiotics_n)
    for feat in top_abx:
        safe_name = feat.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(out_dir, f"10_shap_dependence_{safe_name}.png")
        plot_shap_dependence(shap_exp, feat, out_path)
        print(f"[OK] 10_shap_dependence_{safe_name}.png")

    y_prob_sub = y_prob[sub_idx]
    y_true_sub = np.asarray(y_test)[sub_idx]
    y_pred_sub = (y_prob_sub >= default_threshold).astype(int)
    tn_idx, tp_idx = find_tn_tp_examples(y_true_sub, y_pred_sub, y_prob_sub)

    if tn_idx is not None:
        plot_shap_waterfall_example(shap_exp, tn_idx, os.path.join(out_dir, "11_shap_waterfall_TN_example.png"))
        print(f"[OK] 11_shap_waterfall_TN_example.png")
    else:
        print("[WARN] No clean True-Negative example found in SHAP subsample — skipping.")

    if tp_idx is not None:
        plot_shap_waterfall_example(shap_exp, tp_idx, os.path.join(out_dir, "11_shap_waterfall_TP_example.png"))
        print(f"[OK] 11_shap_waterfall_TP_example.png")
    else:
        print("[WARN] No clean True-Positive example found in SHAP subsample — skipping.")

    # ---- metrics_report.txt (mirrors your original report format) ----
    write_metrics_report(method_name, model, y_test, y_prob, default_threshold, best_t, best_f1,
                          os.path.join(out_dir, "metrics_report.txt"))
    print(f"[OK] metrics_report.txt")
    print(f"[DONE] All visualizations saved to {out_dir}/")


def write_metrics_report(method_name, model, y_true, y_prob, default_threshold, best_f1_threshold, best_f1, out_path):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, balanced_accuracy_score,
        matthews_corrcoef, cohen_kappa_score, classification_report,
    )

    y_pred_default = (y_prob >= default_threshold).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    bal_acc = balanced_accuracy_score(y_true, y_pred_default)
    mcc = matthews_corrcoef(y_true, y_pred_default)
    kappa = cohen_kappa_score(y_true, y_pred_default)

    try:
        best_iter = int(model.best_iteration)
    except Exception:
        best_iter = None

    lines = []
    lines.append(f"FINAL MODEL EVALUATION REPORT — {method_name}")
    lines.append("=" * 50)
    if best_iter is not None:
        lines.append(f"Best Training Iteration: {best_iter}")
    lines.append(f"ROC-AUC:               {roc:.4f}")
    lines.append(f"PR-AUC (AP):           {pr_auc:.4f}")
    lines.append(f"Balanced Accuracy @{default_threshold:.2f}:{bal_acc:.4f}")
    lines.append(f"MCC @{default_threshold:.2f}:              {mcc:.4f}")
    lines.append(f"Cohen's Kappa @{default_threshold:.2f}:    {kappa:.4f}")
    lines.append(f"Best F1 threshold:     {best_f1_threshold:.3f}  (F1={best_f1:.4f})")
    lines.append("")
    lines.append(f"Classification report @ threshold {default_threshold:.2f}:")
    lines.append(classification_report(y_true, y_pred_default))

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
