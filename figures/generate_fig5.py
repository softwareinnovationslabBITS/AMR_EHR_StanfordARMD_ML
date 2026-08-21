import os
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
loss_dir = os.path.join(project_root, "deep_learning/baseline_final_loss_evaluation_v3")
output_dir = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(loss_dir, "baseline_saved_training_history_v3.csv")

df_hist = pd.read_csv(history_path)

RCPARAMS = {
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
}
plt.rcParams.update(RCPARAMS)

TRAIN_COLOR = '#1f77b4'
VAL_COLOR   = '#ff7f0e'
LW          = 2.2

os.makedirs(output_dir, exist_ok=True)


# ── helper ────────────────────────────────────────────────────────────────────
def _save(fig, stem):
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"),          bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {stem}.png / .pdf")


# ── Panel A – Loss ────────────────────────────────────────────────────────────
def plot_loss(ax):
    ax.plot(df_hist['epoch'], df_hist['training_loss'],
            label='Training Loss',   color=TRAIN_COLOR, lw=LW)
    if 'validation_loss' in df_hist.columns:
        ax.plot(df_hist['epoch'], df_hist['validation_loss'],
                label='Validation Loss', color=VAL_COLOR, lw=LW, linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Weighted Binary Cross-Entropy Loss')
    ax.set_title('(A) Training and Validation Loss')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='lightgray')


# ── Panel B – ROC-AUC ────────────────────────────────────────────────────────
def plot_roc_auc(ax):
    if 'training_roc_auc' in df_hist.columns:
        ax.plot(df_hist['epoch'], df_hist['training_roc_auc'],
                label='Training ROC-AUC',   color=TRAIN_COLOR, lw=LW)
    if 'validation_roc_auc' in df_hist.columns:
        ax.plot(df_hist['epoch'], df_hist['validation_roc_auc'],
                label='Validation ROC-AUC', color=VAL_COLOR, lw=LW, linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Area Under ROC Curve (ROC-AUC)')
    ax.set_title('(B) Training and Validation ROC-AUC')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')


# ── Panel C – PR-AUC ─────────────────────────────────────────────────────────
def plot_pr_auc(ax):
    if 'training_pr_auc' in df_hist.columns:
        ax.plot(df_hist['epoch'], df_hist['training_pr_auc'],
                label='Training PR-AUC',   color=TRAIN_COLOR, lw=LW)
    if 'validation_pr_auc' in df_hist.columns:
        ax.plot(df_hist['epoch'], df_hist['validation_pr_auc'],
                label='Validation PR-AUC', color=VAL_COLOR, lw=LW, linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Precision-Recall AUC (PR-AUC / AUPRC)')
    ax.set_title('(C) Training and Validation PR-AUC')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='lightgray')


# ── 1. Individual figures ─────────────────────────────────────────────────────
print("Generating individual Figure 5 panels...")

for stem, plot_fn, caption in [
    (
        "Figure5A_loss",
        plot_loss,
        "Figure 5A. TabTransformer training and validation loss across epochs.\n"
        "Solid blue line = training loss; dashed orange line = validation loss.",
    ),
    (
        "Figure5B_roc_auc",
        plot_roc_auc,
        "Figure 5B. TabTransformer training and validation ROC-AUC across epochs.\n"
        "Solid blue line = training ROC-AUC; dashed orange line = validation ROC-AUC.",
    ),
    (
        "Figure5C_pr_auc",
        plot_pr_auc,
        "Figure 5C. TabTransformer training and validation PR-AUC across epochs.\n"
        "Solid blue line = training PR-AUC; dashed orange line = validation PR-AUC.",
    ),
]:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plot_fn(ax)
    fig.text(0.5, -0.04, caption, ha='center', va='top', fontsize=10)
    fig.tight_layout()
    _save(fig, stem)


# ── 2. Combined figure (unchanged) ───────────────────────────────────────────
print("Generating combined Figure 5...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)
plot_loss(axes[0])
plot_roc_auc(axes[1])
plot_pr_auc(axes[2])

fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.98])
plt.subplots_adjust(wspace=0.28)

caption_text = (
    "Figure 5. TabTransformer training trajectory.\n"
    "(A) Training loss (solid blue line) and validation loss (dashed orange line) across the epochs.\n"
    "(B) Training ROC-AUC (solid blue line) and validation ROC-AUC (dashed orange line) over the training run.\n"
    "(C) Training PR-AUC (solid blue line) and validation PR-AUC (dashed orange line) over the training run."
)
fig.text(0.5, 0.03, caption_text, ha='center', va='center', fontsize=11)

_save(fig, "Figure5_combined")

print("\nDone. Output files:")
for f in ["Figure5A_loss", "Figure5B_roc_auc", "Figure5C_pr_auc", "Figure5_combined"]:
    print(f"  {f}.png / .pdf")
