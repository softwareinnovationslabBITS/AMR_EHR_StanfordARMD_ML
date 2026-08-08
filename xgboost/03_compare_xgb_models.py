"""Comparison plots and manuscript table for completed XGBoost models."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from xgb_config import PLOT_DIR,RESULT_DIR
from xgb_common import ensure_dirs
NAMES={'baseline':'Baseline','class_weighted':'Class weighted','smotenc':'SMOTENC','undersampling':'Undersampling','threshold_optimized':'Threshold optimized','fivefold_cv':'Five-fold CV','optuna':'Optuna tuned'}
def main():
    ensure_dirs(); p=RESULT_DIR/'all_model_results.csv'
    if not p.exists(): raise FileNotFoundError('No model results found')
    df=pd.read_csv(p); df['model']=df.method.map(NAMES).fillna(df.method); df=df.sort_values('validation_pr_auc',ascending=False); df.to_csv(RESULT_DIR/'model_comparison_manuscript_table.csv',index=False)
    metrics=[('test_roc_auc','ROC-AUC'),('test_pr_auc','PR-AUC'),('test_balanced_accuracy','Balanced accuracy'),('test_mcc','MCC')]; y=np.arange(len(df)); offsets=np.linspace(-.24,.24,len(metrics)); fig,ax=plt.subplots(figsize=(10,max(6,.65*len(df))))
    for off,(col,label) in zip(offsets,metrics): ax.scatter(df[col],y+off,s=60,label=label)
    ax.set_yticks(y,df.model); ax.set(xlabel='Test-set performance',title='DL-feature-matched XGBoost model comparison'); ax.grid(axis='x',alpha=.25); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(PLOT_DIR/'xgb_model_comparison_dotplot.png',dpi=300); fig.savefig(PLOT_DIR/'xgb_model_comparison_dotplot.pdf'); plt.close(fig)
    q=df.sort_values('validation_pr_auc'); fig,ax=plt.subplots(figsize=(9,max(6,.55*len(q)))); ax.barh(q.model,q.validation_pr_auc); ax.set(xlabel='Validation PR-AUC',title='Model selection criterion'); ax.grid(axis='x',alpha=.25); fig.tight_layout(); fig.savefig(PLOT_DIR/'xgb_validation_pr_auc_ranking.png',dpi=300); fig.savefig(PLOT_DIR/'xgb_validation_pr_auc_ranking.pdf'); plt.close(fig); print(df.to_string(index=False))
if __name__=='__main__': main()
