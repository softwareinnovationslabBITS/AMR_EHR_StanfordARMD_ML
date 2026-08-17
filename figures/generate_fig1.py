import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

lr_pred_path = os.path.join(project_root, "logistic_regression/logistic_regression_dl_matched_outputs/test_predictions.csv")
xgb_pred_path = os.path.join(project_root, "xgboost/xgb_dl_matched_outputs_v1/predictions/optuna_predictions.npz")
trans_pred_path = os.path.join(project_root, "deep_learning/baseline_ci_bootstrap_results_v2/baseline_test_predictions_v2.npz")
output_dir = os.path.join(project_root, "deep_learning/amr_analysis_outputs")
os.makedirs(output_dir, exist_ok=True)

df_lr = pd.read_csv(lr_pred_path)
y_true_lr, y_prob_lr = df_lr['y_true'].values, df_lr['probability_resistant'].values

xgb_data = np.load(xgb_pred_path)
y_true_xgb, y_prob_xgb = xgb_data['y_test'], xgb_data['test_prob']

trans_data = np.load(trans_pred_path)
y_true_trans, y_prob_trans = trans_data['test_labels'], trans_data['test_probabilities']

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

colors = {'lr': '#1f77b4', 'xgb': '#ff7f0e', 'trans': '#2ca02c'}

# --- Plot A: ROC Curves ---
fig_a, ax_a = plt.subplots(figsize=(7.5, 6.5), dpi=300)
for name, y_true, y_prob, color in [
    ("Logistic Regression", y_true_lr, y_prob_lr, colors['lr']),
    ("Optimized XGBoost", y_true_xgb, y_prob_xgb, colors['xgb']),
    ("TabTransformer", y_true_trans, y_prob_trans, colors['trans'])
]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_a.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {roc_auc:.4f})')

ax_a.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--', label='Random Guess (AUC = 0.50)')
ax_a.set_xlim([-0.01, 1.0])
ax_a.set_ylim([0.0, 1.05])
ax_a.set_xlabel('False Positive Rate (1 - Specificity)')
ax_a.set_ylabel('True Positive Rate (Sensitivity)')
ax_a.set_title('Figure 1A. Receiver Operating Characteristic (ROC) Curves')
ax_a.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='lightgray')
ax_a.grid(True, linestyle=':', alpha=0.6)

fig_a.tight_layout()
fig_a.savefig(os.path.join(output_dir, "Figure1A_roc.png"), bbox_inches='tight')
fig_a.savefig(os.path.join(output_dir, "Figure1A_roc.pdf"), bbox_inches='tight')
plt.close(fig_a)

# --- Plot B: PR Curves ---
fig_b, ax_b = plt.subplots(figsize=(7.5, 6.5), dpi=300)
for name, y_true, y_prob, color in [
    ("Logistic Regression", y_true_lr, y_prob_lr, colors['lr']),
    ("Optimized XGBoost", y_true_xgb, y_prob_xgb, colors['xgb']),
    ("TabTransformer", y_true_trans, y_prob_trans, colors['trans'])
]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    ax_b.plot(recall, precision, color=color, lw=2.5, label=f'{name} (PR-AUC = {pr_auc:.4f})')

prevalence = y_true_lr.mean()
ax_b.axhline(y=prevalence, color='r', lw=1, linestyle='--', label=f'Baseline Prevalence ({prevalence:.3f})')
ax_b.set_xlim([0.0, 1.0])
ax_b.set_ylim([0.0, 1.05])
ax_b.set_xlabel('Recall (Sensitivity)')
ax_b.set_ylabel('Precision (Positive Predictive Value)')
ax_b.set_title('Figure 1B. Precision-Recall (PR) Curves')
ax_b.legend(loc="lower left", bbox_to_anchor=(0.05, 0.22), frameon=True, facecolor='white', edgecolor='lightgray')
ax_b.grid(True, linestyle=':', alpha=0.6)

fig_b.tight_layout()
fig_b.savefig(os.path.join(output_dir, "Figure1B_pr.png"), bbox_inches='tight')
fig_b.savefig(os.path.join(output_dir, "Figure1B_pr.pdf"), bbox_inches='tight')
plt.close(fig_b)

print("Generated Figure 1A and Figure 1B.")
