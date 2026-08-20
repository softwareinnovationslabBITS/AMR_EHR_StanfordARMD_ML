import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shap_dir = os.path.join(project_root, "xgboost/xgb_dl_matched_outputs_v1/shap")
output_dir = os.path.dirname(os.path.abspath(__file__))

npz_path = os.path.join(shap_dir, "best_xgb_shap_values_sample.npz")
data = np.load(npz_path, allow_pickle=True)
values = data['values']
data_values = data['data']
feature_names = data['feature_names']

mean_abs_shap = np.abs(values).mean(axis=0)
TOP_N = 29
sorted_indices = np.argsort(mean_abs_shap)
top_indices = sorted_indices[-TOP_N:]

top_features = [feature_names[i] for i in top_indices]
top_mean_abs = mean_abs_shap[top_indices]
top_shap_values = [values[:, i] for i in top_indices]
top_data_values = [data_values[:, i] for i in top_indices]

shap_colors = ["#008bfb", "#9c59c4", "#ff0051"]
shap_cmap = mcolors.LinearSegmentedColormap.from_list("shap_cmap", shap_colors)

def get_beeswarm_y(x_vals, y_base, width=0.42, max_points_per_bin=20):
    bins = np.linspace(x_vals.min(), x_vals.max(), 100)
    bin_assignments = np.digitize(x_vals, bins)
    y_jitter = np.zeros_like(x_vals)
    for b in np.unique(bin_assignments):
        idx = np.where(bin_assignments == b)[0]
        n_points = len(idx)
        if n_points > 1:
            offsets = np.arange(n_points) - (n_points - 1) / 2.0
            offsets = offsets * (width / max(n_points, max_points_per_bin))
            y_jitter[idx] = offsets
    return y_base + y_jitter

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), gridspec_kw={'width_ratios': [1, 1.35]}, dpi=300)
y_positions = np.arange(TOP_N)

# PANEL A: Bar Chart
bars = ax1.barh(y_positions, top_mean_abs, color="#ff0051", height=0.6, alpha=0.9)
ax1.set_yticks(y_positions)
ax1.set_yticklabels(top_features, fontsize=10.5)
ax1.set_xlabel("Mean absolute SHAP value", fontsize=12, labelpad=10)
ax1.set_title("(A) Feature Importance", fontsize=13, weight='bold', pad=15)
ax1.set_ylim(-0.6, TOP_N - 0.4)

for bar in bars:
    width = bar.get_width()
    ax1.annotate(f'+{width:.2f}',
                 xy=(width, bar.get_y() + bar.get_height() / 2),
                 xytext=(5, 0),
                 textcoords="offset points",
                 ha='left', va='center', fontsize=9, color='#444444', weight='bold')

# PANEL B: Beeswarm Plot
for idx in range(TOP_N):
    f_shap = top_shap_values[idx]
    f_data = top_data_values[idx]
    f_min, f_max = f_data.min(), f_data.max()
    f_norm = (f_data - f_min) / (f_max - f_min) if f_max > f_min else np.zeros_like(f_data)
    y_jittered = get_beeswarm_y(f_shap, y_positions[idx], width=0.45)
    ax2.scatter(f_shap, y_jittered, c=f_norm, cmap=shap_cmap, s=15, alpha=0.8, edgecolors='none', vmin=0, vmax=1)

ax2.axvline(x=0, color='#444444', linestyle='-', alpha=0.6, lw=1.2)
ax2.set_xlabel("SHAP value (impact on model output)", fontsize=12, labelpad=10)
ax2.set_title("(B) Impact on Model Output", fontsize=13, weight='bold', pad=15)
ax2.set_ylim(-0.6, TOP_N - 0.4)
ax2.yaxis.set_tick_params(left=False)
plt.setp(ax2.get_yticklabels(), visible=False)

for ax in [ax1, ax2]:
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    for y in y_positions:
        ax.axhline(y=y, color='gray', linestyle=':', alpha=0.45, lw=0.9)

cbar_ax = fig.add_axes([0.94, 0.25, 0.015, 0.5])
norm = mcolors.Normalize(vmin=0, vmax=1)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=shap_cmap), cax=cbar_ax)
cb.set_ticks([0, 1])
cb.set_ticklabels(['Low', 'High'], fontsize=10)
cb.set_label('Feature value', fontsize=11, labelpad=-5)
cb.ax.tick_params(length=0)
for spine in ['top', 'bottom', 'left', 'right']:
    cbar_ax.spines[spine].set_visible(False)

plt.tight_layout(rect=[0.02, 0.10, 0.91, 0.98])
plt.subplots_adjust(wspace=0.12)

caption_text = (
    "Figure 4. Global SHAP interpretation of the Bayesian-optimized XGBoost model.\n"
    "(A) Mean absolute SHAP value feature importance ranking. (B) SHAP beeswarm plot showing the direction of feature impacts on predictions.\n"
    "Dots are colored by feature value (blue for low, red for high); points to the right of the zero line increase the probability of resistance."
)
fig.text(0.48, 0.04, caption_text, ha='center', va='center', fontsize=11)

os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "Figure4_combined.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, "Figure4_combined.pdf"), bbox_inches='tight')
plt.close()

print("Generated Figure 4.")
