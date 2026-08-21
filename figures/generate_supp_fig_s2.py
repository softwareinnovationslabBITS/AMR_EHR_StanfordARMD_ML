import os
import json
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_dir = os.path.join(project_root, "xgboost", "xgb_dl_matched_outputs_v1")
results_dir = os.path.join(xgb_dir, "results")
pred_dir = os.path.join(xgb_dir, "predictions")
output_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    selection_path = os.path.join(results_dir, "best_xgb_selection.json")
    if not os.path.exists(selection_path):
        raise FileNotFoundError(f"Missing {selection_path}")
        
    with open(selection_path, 'r') as f:
        method = json.load(f)['best_method']
        
    pred_path = os.path.join(pred_dir, f"{method}_predictions.npz")
    z = np.load(pred_path)
    yt = z['y_test'].astype(np.int8)
    tp = z['test_prob'].astype(float)
    
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    ax.hist(tp[yt==0], bins=60, density=True, alpha=0.55, label='Susceptible')
    ax.hist(tp[yt==1], bins=60, density=True, alpha=0.55, label='Resistant')
    
    ax.set(xlabel='Predicted probability of resistance', ylabel='Density', title='Predicted-score distributions')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    
    caption_text = (
        "Supplementary Figure S2. Distribution of predicted probabilities of resistance\n"
        "according to observed susceptibility class."
    )
    fig.text(0.5, -0.06, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S2.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S2.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S2.")

if __name__ == "__main__":
    main()
