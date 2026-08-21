import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ablation_dir = os.path.join(project_root, "deep_learning/amr_ablation_study_outputs")
output_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(ablation_dir, "ablation_summary_metrics.csv")

df = pd.read_csv(summary_path)
cols = list(df.columns)
experiment_col = cols[0]
roc_col = [c for c in cols if 'roc' in c.lower()][0]

control_row = df[df[experiment_col].str.contains('control|none', case=False, na=False)]
control_roc = control_row[roc_col].values[0]

df_ab = df[~df[experiment_col].str.contains('control|none', case=False, na=False)].copy()
df_ab['roc_drop'] = control_roc - df_ab[roc_col]
df_ab = df_ab.sort_values('roc_drop', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
y_pos = np.arange(len(df_ab))
bars = ax.barh(y_pos, df_ab['roc_drop'], color='C0', height=0.6)

ax.set_yticks(y_pos)
labels = df_ab[experiment_col].str.replace('NO_', '').str.replace('_', ' ')
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Performance Drop (Control ROC-AUC - Ablated ROC-AUC)')
ax.set_title('Effect of Feature-Group Removal on TabTransformer Performance', fontsize=13, weight='bold', pad=15)
ax.grid(True, linestyle=':', alpha=0.6, axis='x')

# Add padding to the right so annotations don't overflow the plot area
ax.set_xlim(right=df_ab['roc_drop'].max() * 1.15)

for bar in bars:
    width = bar.get_width()
    ax.annotate(f'{width:.4f}',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha='left', va='center', fontsize=9, weight='bold')

fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])
caption_text = (
    "Figure 6. Effect of feature-group removal on TabTransformer performance.\n"
    "Horizontal bars represent the drop in test-set ROC-AUC when each respective feature category or group is removed relative to the full control model."
)
fig.text(0.5, 0.04, caption_text, ha='center', va='center', fontsize=11)

os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Figure6_combined.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "Figure6_combined.pdf"), bbox_inches='tight')
plt.close()

print("Generated Figure 6.")
