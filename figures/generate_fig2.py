import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define directories
figures_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(figures_dir)
xgb_results_dir = os.path.join(project_root, "xgboost/xgb_dl_matched_outputs_v1/results")
output_dir = os.path.join(project_root, "deep_learning/amr_analysis_outputs")

csv_path = os.path.join(xgb_results_dir, "all_model_results.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Required results file not found: {csv_path}")

df = pd.read_csv(csv_path)

# Map model names to clean display names
NAMES = {
    'baseline': 'Baseline',
    'class_weighted': 'Class weighted',
    'smotenc': 'SMOTENC',
    'undersampling': 'Undersampling',
    'threshold_optimized': 'Threshold optimized',
    'fivefold_cv': 'Five-fold CV',
    'optuna': 'Optuna tuned'
}
df['model'] = df['method'].map(NAMES).fillna(df['method'])
df = df.sort_values('validation_pr_auc', ascending=False)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

metrics = [
    ('test_roc_auc', 'ROC-AUC', '#1f77b4'),
    ('test_pr_auc', 'PR-AUC', '#ff7f0e'),
    ('test_balanced_accuracy', 'Balanced accuracy', '#2ca02c'),
    ('test_mcc', 'MCC', '#d62728')
]

y_pos = np.arange(len(df))
offsets = np.linspace(-0.24, 0.24, len(metrics))

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

for off, (col, label, color) in zip(offsets, metrics):
    ax.scatter(df[col], y_pos + off, s=70, label=label, color=color, alpha=0.9, edgecolor='none')

ax.set_yticks(y_pos)
ax.set_yticklabels(df['model'])
ax.set_xlabel('Test-set performance')
ax.set_title('DL-feature-matched XGBoost model comparison')
ax.grid(axis='x', linestyle=':', alpha=0.5)

# Place legend in bottom left as shown in the original image
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='lightgray', ncol=2)

fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.98])

caption_text = (
    "Figure 2. Performance comparison of DL-feature-matched XGBoost model variations on the independent test set.\n"
    "Metrics shown are Area Under the Receiver Operating Characteristic (ROC-AUC), Precision-Recall Area Under the\n"
    "Curve (PR-AUC), Balanced Accuracy, and Matthews Correlation Coefficient (MCC) across different tuning strategies."
)
fig.text(0.5, 0.04, caption_text, ha='center', va='center', fontsize=11)

os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Figure2_combined.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "Figure2_combined.pdf"), bbox_inches='tight')
plt.close()

print("Generated Figure 2.")
