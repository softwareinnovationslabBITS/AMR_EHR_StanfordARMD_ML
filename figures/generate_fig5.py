import os
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
loss_dir = os.path.join(project_root, "deep_learning/baseline_final_loss_evaluation_v3")
output_dir = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(loss_dir, "baseline_saved_training_history_v3.csv")

df_hist = pd.read_csv(history_path)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)

# (A) Left Plot: Loss
ax_loss = axes[0]
ax_loss.plot(df_hist['epoch'], df_hist['training_loss'], label='Training Loss', color='#1f77b4', lw=2.2)
if 'validation_loss' in df_hist.columns:
    ax_loss.plot(df_hist['epoch'], df_hist['validation_loss'], label='Validation Loss', color='#ff7f0e', lw=2.2, linestyle='--')
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Weighted Binary Cross-Entropy Loss')
ax_loss.set_title('(A) Training and Validation Loss')
ax_loss.grid(True, linestyle=':', alpha=0.6)
ax_loss.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='lightgray')

# (B) Middle Plot: ROC-AUC
ax_auc = axes[1]
if 'training_roc_auc' in df_hist.columns:
    ax_auc.plot(df_hist['epoch'], df_hist['training_roc_auc'], label='Training ROC-AUC', color='#1f77b4', lw=2.2)
if 'validation_roc_auc' in df_hist.columns:
    ax_auc.plot(df_hist['epoch'], df_hist['validation_roc_auc'], label='Validation ROC-AUC', color='#ff7f0e', lw=2.2, linestyle='--')
ax_auc.set_xlabel('Epochs')
ax_auc.set_ylabel('Area Under ROC Curve (ROC-AUC)')
ax_auc.set_title('(B) Training and Validation ROC-AUC')
ax_auc.grid(True, linestyle=':', alpha=0.6)
ax_auc.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')

# (C) Right Plot: PR-AUC
ax_pr = axes[2]
if 'training_pr_auc' in df_hist.columns:
    ax_pr.plot(df_hist['epoch'], df_hist['training_pr_auc'], label='Training PR-AUC', color='#1f77b4', lw=2.2)
if 'validation_pr_auc' in df_hist.columns:
    ax_pr.plot(df_hist['epoch'], df_hist['validation_pr_auc'], label='Validation PR-AUC', color='#ff7f0e', lw=2.2, linestyle='--')
ax_pr.set_xlabel('Epochs')
ax_pr.set_ylabel('Precision-Recall AUC (PR-AUC / AUPRC)')
ax_pr.set_title('(C) Training and Validation PR-AUC')
ax_pr.grid(True, linestyle=':', alpha=0.6)
ax_pr.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')

fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.98])
plt.subplots_adjust(wspace=0.28)

caption_text = (
    "Figure 5. TabTransformer training trajectory.\n"
    "(A) Training loss (solid blue line) and validation loss (dashed orange line) across the 60 epochs.\n"
    "(B) Training ROC-AUC (solid blue line) and validation ROC-AUC (dashed orange line) over the training run.\n"
    "(C) Training PR-AUC (solid blue line) and validation PR-AUC (dashed orange line) over the training run."
)
fig.text(0.5, 0.03, caption_text, ha='center', va='center', fontsize=11)

os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Figure5_combined.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "Figure5_combined.pdf"), bbox_inches='tight')
plt.close()

print("Generated Figure 5 with PR-AUC curve.")
