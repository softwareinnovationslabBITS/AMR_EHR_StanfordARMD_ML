import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_pred_path = os.path.join(project_root, "xgboost/xgb_dl_matched_outputs_v1/predictions/optuna_predictions.npz")
output_dir = os.path.dirname(os.path.abspath(__file__))

xgb_data = np.load(xgb_pred_path)
y_true = xgb_data['y_test']
y_prob = xgb_data['test_prob']
threshold = 0.553

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=300)

# (A) Confusion Matrix
ax_cm = axes[0]
y_pred = (y_prob >= threshold).astype(np.int8)
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

im = ax_cm.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Oranges, vmin=0, vmax=1)
ax_cm.set_title(f'(A) Confusion Matrix (Threshold = {threshold:.3f})')
fig.colorbar(im, ax=ax_cm, label='Proportion of true class')

tick_marks = np.arange(2)
ax_cm.set_xticks(tick_marks)
ax_cm.set_xticklabels(['Susceptible', 'Resistant'])
ax_cm.set_yticks(tick_marks)
ax_cm.set_yticklabels(['Susceptible', 'Resistant'])

for r in range(2):
    for c in range(2):
        count = cm[r, c]
        pct = cm_norm[r, c] * 100
        text_color = "white" if cm_norm[r, c] > 0.5 else "black"
        ax_cm.text(c, r, f"{count:,}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontsize=10, weight='bold')

ax_cm.set_ylabel('True Class')
ax_cm.set_xlabel('Predicted Class')

# (B) Calibration Curve
ax_cal = axes[1]
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy='quantile')
ax_cal.plot(prob_pred, prob_true, marker='o', color='C1', lw=2, label='XGBoost (Optuna)')
ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
ax_cal.set_xlabel('Mean Predicted Probability')
ax_cal.set_ylabel('Observed Resistant Proportion')
ax_cal.set_title('(B) Calibration Curve')
ax_cal.legend(loc="upper left", frameon=True)
ax_cal.grid(True, linestyle=':', alpha=0.6)

fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.98])
plt.subplots_adjust(wspace=0.32)

caption_text = (
    "Figure 3. Independent test-set performance of the Bayesian-optimized XGBoost model.\n"
    "(A) Confusion matrix at the validation-selected threshold of 0.553. (B) Calibration curve."
)
fig.text(0.5, 0.04, caption_text, ha='center', va='center', fontsize=11)

os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Figure3_combined.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "Figure3_combined.pdf"), bbox_inches='tight')
plt.close()

print("Generated Figure 3.")
