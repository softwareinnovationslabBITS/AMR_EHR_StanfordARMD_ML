import os
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ablation_dir = os.path.join(project_root, "deep_learning", "amr_ablation_study_outputs")
output_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    history_path = os.path.join(ablation_dir, "ablation_training_history.csv")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Missing {history_path}")
        
    history = pd.read_csv(history_path)
    
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Plot validation ROC-AUC trajectory for each experiment
    for exp in history["experiment"].unique():
        exp_data = history[history["experiment"] == exp]
        ax.plot(exp_data["epoch"], exp_data["validation_roc_auc"], label=exp, linewidth=1.5)
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Ablation Training Curves")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, ncol=2)
    
    caption_text = (
        "Supplementary Figure S8. Validation ROC-AUC trajectories during TabTransformer\n"
        "feature-group ablation training."
    )
    fig.text(0.5, -0.05, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    # Increase right margin so legend fits
    fig.tight_layout(rect=[0, 0.05, 0.7, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S8.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S8.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S8.")

if __name__ == "__main__":
    main()
