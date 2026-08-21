import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_dir = os.path.join(project_root, "xgboost", "xgb_dl_matched_outputs_v1")
shap_dir = os.path.join(xgb_dir, "shap")
output_dir = os.path.dirname(os.path.abspath(__file__))

# The exact 5 features from the S5 screenshot
FEATURES = [
    "antibiotic_enc",
    "order_month",
    "order_year",
    "organism_enc",
    "prior_resistance_count"
]

def main():
    npz_path = os.path.join(shap_dir, "best_xgb_shap_values_sample.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}")
        
    data = np.load(npz_path, allow_pickle=True)
    values = data['values']
    base_values = data['base_values']
    data_values = data['data']
    feature_names = data['feature_names']
    
    # Reconstruct Explanation object
    ex = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data_values,
        feature_names=feature_names.tolist()
    )
    
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(3, 2, figsize=(14, 18), dpi=300)
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        try:
            shap.plots.scatter(ex[:, feature], ax=ax, show=False)
        except Exception as e:
            # Fallback if ax is not supported in the user's SHAP version
            print(f"Warning plotting {feature}: {e}. Skipping subplot.")
            
    # Hide the empty 6th subplot
    axes[5].axis('off')
    
    caption_text = (
        "Supplementary Figure S5. Selected SHAP dependence plots for the Bayesian-optimized XGBoost model."
    )
    fig.text(0.5, 0.08, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S5.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S5.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S5.")

if __name__ == "__main__":
    main()
