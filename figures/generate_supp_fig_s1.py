import os
import json
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_dir = os.path.join(project_root, "xgboost", "xgb_dl_matched_outputs_v1")
results_dir = os.path.join(xgb_dir, "results")
output_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    curve_path = os.path.join(results_dir, "best_xgb_validation_threshold_curve.csv")
    selection_path = os.path.join(results_dir, "best_xgb_selection.json")
    
    if not os.path.exists(curve_path):
        raise FileNotFoundError(f"Missing {curve_path}")
        
    q = pd.read_csv(curve_path)
    
    with open(selection_path, 'r') as f:
        selection = json.load(f)
    t = selection['threshold']

    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    for c in ['precision', 'recall', 'f1', 'mcc', 'balanced_accuracy']: 
        ax.plot(q['threshold'], q[c], label=c.replace('_', ' ').title())
        
    ax.axvline(t, ls='--', label=f'Selected={t:.3f}')
    ax.set(xlabel='Decision threshold', ylabel='Validation metric', title='Validation threshold analysis')
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 0.02), frameon=False, ncol=2)
    ax.grid(alpha=.25)
    
    caption_text = (
        "Supplementary Figure S1. Validation threshold analysis for the Bayesian-optimized XGBoost model."
    )
    fig.text(0.5, -0.05, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S1.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S1.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S1.")

if __name__ == "__main__":
    main()
