import os
import numpy as np
import matplotlib.pyplot as plt
import shap

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_dir = os.path.join(project_root, "xgboost", "xgb_dl_matched_outputs_v1")
shap_dir = os.path.join(xgb_dir, "shap")
output_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    npz_path = os.path.join(shap_dir, "best_xgb_shap_values_sample.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}")
        
    data = np.load(npz_path, allow_pickle=True)
    values = data['values']
    base_values = data['base_values']
    data_values = data['data']
    feature_names = data['feature_names']
    
    ex = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data_values,
        feature_names=feature_names.tolist()
    )
    
    # Calculate predicted probabilities to find max and min
    # Note: SHAP outputs log-odds for XGBoost, max log-odds = max probability
    log_odds = base_values + values.sum(axis=1)
    high_idx = int(np.argmax(log_odds))
    low_idx = int(np.argmin(log_odds))
    
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=(10, 14), dpi=300)
    
    # We use plt.subplot because shap.plots.waterfall plots to current active axis
    plt.subplot(2, 1, 1)
    shap.plots.waterfall(ex[high_idx], max_display=20, show=False)
    
    plt.subplot(2, 1, 2)
    shap.plots.waterfall(ex[low_idx], max_display=20, show=False)
    
    caption_text = (
        "Supplementary Figure S6. Representative SHAP waterfall plots for high- and\n"
        "low-probability predictions."
    )
    fig.text(0.5, 0.02, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S6.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S6.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S6.")

if __name__ == "__main__":
    main()
