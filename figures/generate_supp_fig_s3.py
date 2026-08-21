import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xgb_dir = os.path.join(project_root, "xgboost", "xgb_dl_matched_outputs_v1")
results_dir = os.path.join(xgb_dir, "results")
output_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    ci_path = os.path.join(results_dir, "best_xgb_bootstrap_confidence_intervals.csv")
    if not os.path.exists(ci_path):
        raise FileNotFoundError(f"Missing {ci_path}")
        
    ci = pd.read_csv(ci_path)
    
    # Same logic as forest() in analyze_best_xgb.py
    q = ci.copy()
    q['display'] = q.metric.replace({
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall_sensitivity': 'Recall/sensitivity',
        'specificity': 'specificity',
        'negative_predictive_value': 'Negative predictive value',
        'f1': 'F1-score',
        'roc_auc': 'ROC-AUC',
        'pr_auc': 'PR-AUC',
        'balanced_accuracy': 'Balanced accuracy',
        'mcc': 'MCC',
        'cohen_kappa': "Cohen's kappa",
        'brier_score': 'Brier score'
    })
    # Reverse to match order
    q = q.iloc[::-1]
    
    x = q.point_estimate.to_numpy()
    lo = x - q.ci_lower_95.to_numpy()
    hi = q.ci_upper_95.to_numpy() - x
    
    method = str(ci['method'].iloc[0])
    
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, max(6, 0.42 * len(q))), dpi=300)
    
    ax.errorbar(x, np.arange(len(q)), xerr=np.vstack([lo, hi]), fmt='o', capsize=3)
    ax.set_yticks(np.arange(len(q)))
    ax.set_yticklabels(q.display)
    
    ax.set(xlabel='Point estimate and 95% bootstrap CI', title=f'Best feature-matched XGBoost metrics — {method}')
    ax.grid(axis='x', alpha=0.25)
    
    caption_text = (
        "Supplementary Figure S3. Bootstrap confidence intervals for test-set performance\n"
        "metrics of the Bayesian-optimized XGBoost model."
    )
    fig.text(0.5, -0.05, caption_text, ha='center', va='top', fontsize=12, weight='bold')
    
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S3.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, "Supplementary_Figure_S3.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("Generated Supplementary Figure S3.")

if __name__ == "__main__":
    main()
