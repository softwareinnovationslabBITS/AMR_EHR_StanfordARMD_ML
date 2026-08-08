"""Bootstrap CIs and full diagnostic/SHAP analysis for best validation PR-AUC model."""
from __future__ import annotations
import json,time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,shap,xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve,roc_curve
from xgb_config import *
from xgb_common import *

def bootstrap(y,p,t):
    rng=np.random.default_rng(RANDOM_SEED); neg=np.flatnonzero(y==0); pos=np.flatnonzero(y==1)
    names=['accuracy','precision','recall_sensitivity','specificity','negative_predictive_value','f1','roc_auc','pr_auc','balanced_accuracy','mcc','cohen_kappa','brier_score']
    dist={n:np.empty(N_BOOTSTRAPS) for n in names}; start=time.time()
    for i in range(N_BOOTSTRAPS):
        idx=np.concatenate([rng.choice(neg,len(neg),replace=True),rng.choice(pos,len(pos),replace=True)]); rng.shuffle(idx); m=metrics(y[idx],p[idx],t)
        for n in names: dist[n][i]=m[n]
        if (i+1)%100==0 or i+1==N_BOOTSTRAPS: print(f'Bootstrap {i+1}/{N_BOOTSTRAPS}; {(time.time()-start)/60:.1f} min')
    ddf=pd.DataFrame(dist); ddf.insert(0,'bootstrap_iteration',np.arange(1,N_BOOTSTRAPS+1)); point=metrics(y,p,t); alpha=1-BOOTSTRAP_CONFIDENCE; rows=[]
    for n in names:
        lo,hi=np.percentile(dist[n],[100*alpha/2,100*(1-alpha/2)])
        rows.append({'metric':n,'point_estimate':point[n],'bootstrap_mean':float(np.mean(dist[n])),'bootstrap_standard_error':float(np.std(dist[n],ddof=1)),
                     'ci_lower_95':float(lo),'ci_upper_95':float(hi),'formatted':f"{point[n]:.4f} ({lo:.4f}–{hi:.4f})",'n_bootstraps':N_BOOTSTRAPS,
                     'bootstrap_method':'stratified_record_level_percentile','threshold':t,'threshold_selected_on':'validation'})
    return ddf,pd.DataFrame(rows)

def savefig(fig,name): fig.tight_layout(); fig.savefig(PLOT_DIR/f'{name}.png',dpi=300,bbox_inches='tight'); fig.savefig(PLOT_DIR/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)

def standard_plots(yv,vp,yt,tp,t,method):
    fpr,tpr,_=roc_curve(yt,tp); fig,ax=plt.subplots(figsize=(7,6)); ax.plot(fpr,tpr,lw=2,label=f"ROC-AUC = {metrics(yt,tp,t)['roc_auc']:.4f}"); ax.plot([0,1],[0,1],'--'); ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title=f'ROC curve — {method}'); ax.legend(frameon=False); ax.grid(alpha=.25); savefig(fig,'best_xgb_01_roc_curve')
    pr,rc,_=precision_recall_curve(yt,tp); fig,ax=plt.subplots(figsize=(7,6)); ax.plot(rc,pr,lw=2,label=f'PR-AUC = {average_precision_score(yt,tp):.4f}'); ax.axhline(yt.mean(),ls='--',label=f'Prevalence = {yt.mean():.3f}'); ax.set(xlabel='Recall',ylabel='Precision',title=f'Precision–recall curve — {method}'); ax.legend(frameon=False); ax.grid(alpha=.25); savefig(fig,'best_xgb_02_pr_curve')
    pred=(tp>=t).astype(np.int8); cm=confusion_matrix(yt,pred,labels=[0,1]); norm=cm/cm.sum(axis=1,keepdims=True); fig,ax=plt.subplots(figsize=(6.5,5.5)); im=ax.imshow(norm)
    for i in range(2):
        for j in range(2): ax.text(j,i,f'{cm[i,j]:,}\n({norm[i,j]:.3f})',ha='center',va='center')
    ax.set_xticks([0,1],['Susceptible','Resistant']); ax.set_yticks([0,1],['Susceptible','Resistant']); ax.set(xlabel='Predicted class',ylabel='True class',title=f'Confusion matrix — {method}\nThreshold={t:.3f}'); fig.colorbar(im,ax=ax,label='Row proportion'); savefig(fig,'best_xgb_03_confusion_matrix')
    rows=[]
    for th in np.linspace(.001,.999,THRESHOLD_GRID_SIZE):
        m=metrics(yv,vp,th); rows.append({'threshold':th,'precision':m['precision'],'recall':m['recall_sensitivity'],'f1':m['f1'],'mcc':m['mcc'],'balanced_accuracy':m['balanced_accuracy']})
    q=pd.DataFrame(rows); q.to_csv(RESULT_DIR/'best_xgb_validation_threshold_curve.csv',index=False); fig,ax=plt.subplots(figsize=(9,6))
    for c in ['precision','recall','f1','mcc','balanced_accuracy']: ax.plot(q.threshold,q[c],label=c.replace('_',' ').title())
    ax.axvline(t,ls='--',label=f'Selected={t:.3f}'); ax.set(xlabel='Decision threshold',ylabel='Validation metric',title='Validation threshold analysis'); ax.legend(frameon=False,ncol=2); ax.grid(alpha=.25); savefig(fig,'best_xgb_04_threshold_analysis')
    frac,mean=calibration_curve(yt,tp,n_bins=15,strategy='quantile'); fig,ax=plt.subplots(figsize=(7,6)); ax.plot(mean,frac,marker='o',label=method); ax.plot([0,1],[0,1],'--',label='Perfect calibration'); ax.set(xlabel='Mean predicted probability',ylabel='Observed resistant proportion',title='Calibration curve'); ax.legend(frameon=False); ax.grid(alpha=.25); savefig(fig,'best_xgb_05_calibration_curve')
    fig,ax=plt.subplots(figsize=(8,6)); ax.hist(tp[yt==0],bins=60,density=True,alpha=.55,label='Susceptible'); ax.hist(tp[yt==1],bins=60,density=True,alpha=.55,label='Resistant'); ax.set(xlabel='Predicted probability of resistance',ylabel='Density',title='Predicted-score distributions'); ax.legend(frameon=False); ax.grid(alpha=.2); savefig(fig,'best_xgb_06_score_distribution')

def importance_and_shap(model,X):
    score=model.get_booster().get_score(importance_type='gain'); imp=pd.DataFrame([{'feature':k,'gain':v} for k,v in score.items()]).sort_values('gain',ascending=False); imp.to_csv(RESULT_DIR/'best_xgb_gain_feature_importance.csv',index=False)
    top=imp.head(TOP_N_FEATURES).sort_values('gain'); fig,ax=plt.subplots(figsize=(9,max(6,.28*len(top)))); ax.barh(top.feature,top.gain); ax.set(xlabel='Mean gain',title=f'Top {TOP_N_FEATURES} XGBoost features by gain'); ax.grid(axis='x',alpha=.25); savefig(fig,'best_xgb_07_gain_feature_importance')
    rng=np.random.default_rng(RANDOM_SEED); idx=rng.choice(len(X),size=min(SHAP_SAMPLE_SIZE,len(X)),replace=False); Xs=X.iloc[idx].copy(); Xnumeric=Xs.copy()
    for c in Xnumeric.select_dtypes(include='category').columns: Xnumeric[c]=Xnumeric[c].cat.codes.astype(np.int32)
    explainer=shap.TreeExplainer(model); ex=explainer(Xnumeric)
    np.savez_compressed(SHAP_DIR/'best_xgb_shap_values_sample.npz',values=np.asarray(ex.values,dtype=np.float32),base_values=np.asarray(ex.base_values,dtype=np.float32),data=Xnumeric.to_numpy(dtype=np.float32),feature_names=np.asarray(Xnumeric.columns,dtype=object),sampled_test_indices=idx)
    plt.figure(); shap.plots.beeswarm(ex,max_display=TOP_N_FEATURES,show=False); plt.tight_layout(); plt.savefig(SHAP_DIR/'best_xgb_08_shap_beeswarm.png',dpi=300,bbox_inches='tight'); plt.savefig(SHAP_DIR/'best_xgb_08_shap_beeswarm.pdf',bbox_inches='tight'); plt.close()
    plt.figure(); shap.plots.bar(ex,max_display=TOP_N_FEATURES,show=False); plt.tight_layout(); plt.savefig(SHAP_DIR/'best_xgb_09_shap_mean_abs_bar.png',dpi=300,bbox_inches='tight'); plt.savefig(SHAP_DIR/'best_xgb_09_shap_mean_abs_bar.pdf',bbox_inches='tight'); plt.close()
    simp=pd.DataFrame({'feature':Xnumeric.columns,'mean_absolute_shap':np.abs(ex.values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False); simp.to_csv(SHAP_DIR/'best_xgb_shap_importance.csv',index=False)
    for feature in simp.head(TOP_N_DEPENDENCE).feature:
        plt.figure(); shap.plots.scatter(ex[:,feature],show=False); plt.tight_layout(); safe=''.join(c if c.isalnum() or c in '-_' else '_' for c in feature); plt.savefig(SHAP_DIR/f'best_xgb_10_shap_dependence_{safe}.png',dpi=300,bbox_inches='tight'); plt.close()
    probs=model.predict_proba(Xs)[:,1]
    for label,i in {'high_probability':int(np.argmax(probs)),'low_probability':int(np.argmin(probs))}.items():
        plt.figure(); shap.plots.waterfall(ex[i],max_display=20,show=False); plt.tight_layout(); plt.savefig(SHAP_DIR/f'best_xgb_11_shap_waterfall_{label}.png',dpi=300,bbox_inches='tight'); plt.close()

def forest(ci,method):
    q=ci.copy(); q['display']=q.metric.replace({'recall_sensitivity':'Recall/sensitivity','negative_predictive_value':'Negative predictive value','balanced_accuracy':'Balanced accuracy','cohen_kappa':"Cohen's kappa",'brier_score':'Brier score','roc_auc':'ROC-AUC','pr_auc':'PR-AUC','mcc':'MCC','f1':'F1-score'}); q=q.iloc[::-1]; x=q.point_estimate.to_numpy(); lo=x-q.ci_lower_95.to_numpy(); hi=q.ci_upper_95.to_numpy()-x
    fig,ax=plt.subplots(figsize=(9,max(6,.42*len(q)))); ax.errorbar(x,np.arange(len(q)),xerr=np.vstack([lo,hi]),fmt='o',capsize=3); ax.set_yticks(np.arange(len(q)),q.display); ax.set(xlabel='Point estimate and 95% bootstrap CI',title=f'Best feature-matched XGBoost metrics — {method}'); ax.grid(axis='x',alpha=.25); savefig(fig,'best_xgb_12_bootstrap_ci_forest')

def main():
    ensure_dirs(); rp=RESULT_DIR/'all_model_results.csv'
    if not rp.exists(): raise FileNotFoundError('Run 01_train_xgb_variations.py first')
    res=pd.read_csv(rp).dropna(subset=['validation_pr_auc']); best=res.sort_values('validation_pr_auc',ascending=False).iloc[0]; method=str(best.method); print(f'Best by validation PR-AUC: {method}')
    z=np.load(PRED_DIR/f'{method}_predictions.npz'); yv=z['y_val'].astype(np.int8); vp=z['val_prob'].astype(float); yt=z['y_test'].astype(np.int8); tp=z['test_prob'].astype(float); t,ts=select_threshold(yv,vp)
    pd.DataFrame([{'method':method,'selected_by':'validation_pr_auc','validation_pr_auc':float(best.validation_pr_auc),**metrics(yt,tp,t)}]).to_csv(RESULT_DIR/'best_xgb_test_point_metrics.csv',index=False)
    dist,ci=bootstrap(yt,tp,t); dist.to_csv(RESULT_DIR/'best_xgb_bootstrap_distributions.csv.gz',index=False,compression='gzip'); ci.insert(0,'method',method); ci.to_csv(RESULT_DIR/'best_xgb_bootstrap_confidence_intervals.csv',index=False)
    model=xgb.XGBClassifier(); model.load_model(MODEL_DIR/method/'model.ubj'); data=arrays_to_frames(load_bundle()); standard_plots(yv,vp,yt,tp,t,method); importance_and_shap(model,data['X_test']); forest(ci,method)
    with (RESULT_DIR/'best_xgb_selection.json').open('w') as f: json.dump({'best_method':method,'selected_by':'highest_validation_pr_auc','validation_pr_auc':float(best.validation_pr_auc),'threshold':t,'threshold_selected_on':'validation','threshold_objective':THRESHOLD_OBJECTIVE,'validation_threshold_score':ts,'bootstrap_repetitions':N_BOOTSTRAPS},f,indent=2)
    print(ci[['metric','formatted']].to_string(index=False))
if __name__=='__main__': main()
