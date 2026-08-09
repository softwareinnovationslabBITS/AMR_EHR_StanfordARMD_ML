# Source: /AMR_Stanford/DL_codes/amr_project/xgb_dl_feature_matched_project/train_xgb_variations.py
"""Train baseline, imbalance, 5-fold CV and Optuna XGBoost variants."""
from __future__ import annotations
import gc,time
import numpy as np,pandas as pd,optuna
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgb_config import *
from xgb_common import *

def fit_score(method,model,Xfit,yfit,d,threshold_mode='validation',extra=None):
    t0=time.time(); model.fit(Xfit,yfit,eval_set=[(d['X_val'],d['y_val'])],verbose=False); elapsed=time.time()-t0
    vp=predict_prob(model,d['X_val']); tp=predict_prob(model,d['X_test'])
    if threshold_mode=='validation': th,score=select_threshold(d['y_val'],vp)
    else: th,score=.5,np.nan
    vm=metrics(d['y_val'],vp,th); tm=metrics(d['y_test'],tp,th)
    row={'method':method,'validation_pr_auc':vm['pr_auc'],'validation_roc_auc':vm['roc_auc'],'validation_mcc':vm['mcc'],
         'threshold':th,'test_roc_auc':tm['roc_auc'],'test_pr_auc':tm['pr_auc'],'test_accuracy':tm['accuracy'],
         'test_precision':tm['precision'],'test_recall_sensitivity':tm['recall_sensitivity'],'test_specificity':tm['specificity'],
         'test_f1':tm['f1'],'test_balanced_accuracy':tm['balanced_accuracy'],'test_mcc':tm['mcc'],
         'test_cohen_kappa':tm['cohen_kappa'],'test_brier_score':tm['brier_score'],
         'best_iteration':int(getattr(model,'best_iteration',-1)),'elapsed_seconds':elapsed}
    meta={'method':method,'threshold':th,'threshold_selected_on':'validation' if threshold_mode=='validation' else 'fixed_0.5',
          'threshold_objective':THRESHOLD_OBJECTIVE if threshold_mode=='validation' else None,
          'validation_threshold_score':score,'validation_metrics':vm,'test_metrics':tm,
          'best_iteration':int(getattr(model,'best_iteration',-1)),'elapsed_seconds':elapsed,'model_parameters':model.get_params()}
    if extra: meta.update(extra)
    save_run(method,model,meta,vp,tp,d['y_val'],d['y_test']); append_result(row)
    print(f"[{method}] Val PR-AUC={vm['pr_auc']:.4f} | Test PR-AUC={tm['pr_auc']:.4f} | ROC-AUC={tm['roc_auc']:.4f} | MCC={tm['mcc']:.4f}")
    return row

def baseline(d): return fit_score('baseline',make_model(),d['X_train'],d['y_train'],d,'fixed')
def weighted(d):
    r=ratio(d['y_train']); return fit_score('class_weighted',make_model({'scale_pos_weight':r}),d['X_train'],d['y_train'],d,'fixed',{'scale_pos_weight':r})
def thresholded(d):
    r=ratio(d['y_train']); return fit_score('threshold_optimized',make_model({'scale_pos_weight':r}),d['X_train'],d['y_train'],d,'validation',{'scale_pos_weight':r})

def smotenc(d):
    if not RUN_SMOTENC: print('[smotenc] skipped'); return
    catidx=[d['all_features'].index(c) for c in d['cat_features']+d['binary_features']]
    sampler=SMOTENC(categorical_features=catidx,sampling_strategy=SMOTENC_SAMPLING_STRATEGY,
                    random_state=RANDOM_SEED,k_neighbors=SMOTENC_K_NEIGHBORS)
    X=d['X_train'].to_numpy(dtype=np.float32); y=d['y_train']; t0=time.time(); Xr,yr=sampler.fit_resample(X,y); sec=time.time()-t0
    Xrdf=restore_frame(Xr,d)
    out=fit_score('smotenc',make_model(),Xrdf,np.asarray(yr,dtype=np.int8),d,'fixed',
                  {'sampling_strategy':SMOTENC_SAMPLING_STRATEGY,'rows_before':len(y),'rows_after':len(yr),'sampling_seconds':sec})
    del X,Xr,yr,Xrdf; gc.collect(); return out

def undersample(d):
    if not RUN_UNDERSAMPLING: print('[undersampling] skipped'); return
    sampler=RandomUnderSampler(sampling_strategy=UNDERSAMPLING_STRATEGY,random_state=RANDOM_SEED)
    X=d['X_train'].to_numpy(dtype=np.float32); y=d['y_train']; t0=time.time(); Xr,yr=sampler.fit_resample(X,y); sec=time.time()-t0
    Xrdf=restore_frame(Xr,d)
    out=fit_score('undersampling',make_model(),Xrdf,np.asarray(yr,dtype=np.int8),d,'fixed',
                  {'sampling_strategy':UNDERSAMPLING_STRATEGY,'rows_before':len(y),'rows_after':len(yr),'sampling_seconds':sec})
    del X,Xr,yr,Xrdf; gc.collect(); return out

def fivefold(d):
    skf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=RANDOM_SEED); rows=[]; X=d['X_train']; y=d['y_train']; t0=time.time()
    for fold,(tr,va) in enumerate(skf.split(X,y),1):
        m=make_model(); m.fit(X.iloc[tr],y[tr],eval_set=[(X.iloc[va],y[va])],verbose=False)
        p=predict_prob(m,X.iloc[va]); th,_=select_threshold(y[va],p); mm=metrics(y[va],p,th); mm['fold']=fold; mm['best_iteration']=int(getattr(m,'best_iteration',-1)); rows.append(mm)
        print(f"[CV {fold}/{N_FOLDS}] PR-AUC={mm['pr_auc']:.4f} ROC-AUC={mm['roc_auc']:.4f} MCC={mm['mcc']:.4f}")
        del m,p; gc.collect()
    f=pd.DataFrame(rows); f.to_csv(RESULT_DIR/'fivefold_cv_fold_metrics.csv',index=False)
    summary={c:{'mean':float(f[c].mean()),'std':float(f[c].std(ddof=1))} for c in ['roc_auc','pr_auc','balanced_accuracy','mcc','cohen_kappa','f1']}
    return fit_score('fivefold_cv',make_model(),X,y,d,'validation',{'n_folds':N_FOLDS,'cv_summary':summary,'cv_seconds':time.time()-t0})

def objective(trial,d):
    r=ratio(d['y_train']); params={'max_depth':trial.suggest_int('max_depth',3,10),
      'learning_rate':trial.suggest_float('learning_rate',.005,.12,log=True),'subsample':trial.suggest_float('subsample',.6,1),
      'colsample_bytree':trial.suggest_float('colsample_bytree',.5,1),'min_child_weight':trial.suggest_float('min_child_weight',1,30,log=True),
      'gamma':trial.suggest_float('gamma',1e-8,5,log=True),'reg_alpha':trial.suggest_float('reg_alpha',1e-8,20,log=True),
      'reg_lambda':trial.suggest_float('reg_lambda',1e-3,30,log=True),'scale_pos_weight':trial.suggest_float('scale_pos_weight',r*.5,r*1.5),
      'n_estimators':1800,'early_stopping_rounds':50}
    skf=StratifiedKFold(n_splits=OPTUNA_FOLDS,shuffle=True,random_state=RANDOM_SEED); scores=[]; X=d['X_train']; y=d['y_train']
    for fold,(tr,va) in enumerate(skf.split(X,y),1):
        m=make_model(params); m.fit(X.iloc[tr],y[tr],eval_set=[(X.iloc[va],y[va])],verbose=False); p=predict_prob(m,X.iloc[va]); scores.append(float(average_precision_score(y[va],p)))
        trial.report(float(np.mean(scores)),step=fold)
        if trial.should_prune(): raise optuna.TrialPruned()
        del m,p; gc.collect()
    return float(np.mean(scores))

def optuna_train(d):
    storage=f"sqlite:///{(OPTUNA_DIR/'study.db').resolve()}"; study=optuna.create_study(study_name='xgb_dl_matched_pr_auc',storage=storage,load_if_exists=True,direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1))
    t0=time.time(); study.optimize(lambda trial:objective(trial,d),n_trials=N_OPTUNA_TRIALS,timeout=OPTUNA_TIMEOUT_SECONDS,gc_after_trial=True,show_progress_bar=False)
    study.trials_dataframe().to_csv(OPTUNA_DIR/'optuna_trials.csv',index=False)
    params=dict(study.best_params); params.update({'n_estimators':2500,'early_stopping_rounds':75})
    return fit_score('optuna',make_model(params),d['X_train'],d['y_train'],d,'validation',
                     {'optuna_best_cv_pr_auc':float(study.best_value),'optuna_best_params':study.best_params,'optuna_folds':OPTUNA_FOLDS,'search_seconds':time.time()-t0})

TRAIN={'baseline':baseline,'class_weighted':weighted,'smotenc':smotenc,'undersampling':undersample,'threshold_optimized':thresholded,'fivefold_cv':fivefold,'optuna':optuna_train}

def main():
    ensure_dirs(); set_seed(); print('Loading exact DL arrays...'); d=arrays_to_frames(load_bundle())
    print(f"Train={len(d['y_train']):,} Val={len(d['y_val']):,} Test={len(d['y_test']):,} Features={len(d['all_features'])}")
    for method in MODEL_SEQUENCE:
        if complete(method): print(f'[{method}] complete artifacts exist; skipping'); continue
        print('\n'+'='*80+f'\nRUNNING {method}\n'+'='*80)
        try: TRAIN[method](d)
        except Exception as e: print(f'[ERROR] {method}: {type(e).__name__}: {e}')
    p=RESULT_DIR/'all_model_results.csv'
    if p.exists():
        df=pd.read_csv(p).sort_values('validation_pr_auc',ascending=False); df.to_csv(RESULT_DIR/'all_model_results_ranked.csv',index=False); print(df.to_string(index=False))
if __name__=='__main__': main()
