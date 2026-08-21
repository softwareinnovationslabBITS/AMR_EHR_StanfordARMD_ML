import os
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_dir = os.path.join(project_root, "xgboost", "xgb_dl_matched_outputs_v1")
results_dir = os.path.join(xgb_dir, "results")
output_dir = os.path.dirname(os.path.abspath(__file__))

TOP_N_FEATURES = 30

def main():
    imp_path = os.path.join(results_dir, "best_xgb_gain_feature_importance.csv")
    if not os.path.exists(imp_path):
        raise FileNotFoundError(f"Missing {imp_path}")
        
    imp = pd.read_csv(imp_path)
    
    top = imp.head(TOP_N_FEATURES).sort_values('gain')
    
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, max(6, 0.28 * len(top))), dpi=300)
    
    ax.barh(top.feature, top.gain)
    ax.set(xlabel='Mean gain', title=f'Top {TOP_N_FEATURES} XGBoost features by gain')
    ax.grid(axis='x', alpha=0.25)
    
    caption_text = (
        "Supplementary Figure S4. XGBoost feature importance based on mean gain."
    )
    fig.text(0.5, -0.05, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S4.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S4.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S4.")

if __name__ == "__main__":
    main()
