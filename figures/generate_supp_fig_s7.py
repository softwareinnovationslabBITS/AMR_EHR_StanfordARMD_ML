import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ablation_dir = os.path.join(project_root, "deep_learning", "amr_ablation_study_outputs")
output_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    summary_path = os.path.join(ablation_dir, "ablation_summary_metrics.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing {summary_path}")
        
    summary = pd.read_csv(summary_path)
    
    plot_data = summary.sort_values("test_roc_auc", ascending=True).copy()
    y_positions = np.arange(len(plot_data))
    
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(10, max(6, 0.48 * len(plot_data))), dpi=300)
    
    ax.plot(plot_data["test_roc_auc"], y_positions, "o-", label="ROC-AUC")
    ax.plot(plot_data["test_pr_auc"], y_positions, "s-", label="PR-AUC")
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["experiment"])
    ax.set_xlabel("Test-set score")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=True)
    
    caption_text = (
        "Supplementary Figure S7. Test ROC-AUC and PR-AUC across TabTransformer\n"
        "feature-group ablation experiments."
    )
    fig.text(0.5, -0.05, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S7.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S7.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S7.")

if __name__ == "__main__":
    main()
