"""Shared helpers for DL-feature-matched XGBoost workflow."""
from __future__ import annotations
import gc, json, random
from typing import Any, Dict, List, Sequence, Optional, Tuple
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from xgb_config import *

def ensure_dirs():
    for p in [OUT,MODEL_DIR,PRED_DIR,RESULT_DIR,PLOT_DIR,SHAP_DIR,OPTUNA_DIR,LOG_DIR]: p.mkdir(parents=True,exist_ok=True)

def set_seed(seed=RANDOM_SEED): random.seed(seed); np.random.seed(seed)

def load_bundle():
    if not DL_BUNDLE_PATH.exists(): raise FileNotFoundError(f'Missing {DL_BUNDLE_PATH.resolve()}')
    b=joblib.load(DL_BUNDLE_PATH)
    req={'CAT_FEATURES','CONT_FEATURES','BINARY_FEATURES','ALL_FEATURES','X_train','X_val','X_test','y_train','y_val','y_test'}
    miss=req-set(b)
    if miss: raise KeyError(f'Missing bundle keys: {sorted(miss)}')
    return b

def arrays_to_frames(b):
    cat=list(b['CAT_FEATURES']); cont=list(b['CONT_FEATURES']); binary=list(b['BINARY_FEATURES']); allf=list(b['ALL_FEATURES'])
    arrays={k:np.asarray(b[k],dtype=np.float32) for k in ['X_train','X_val','X_test']}
    ys={k:np.asarray(b[k],dtype=np.int8).ravel() for k in ['y_train','y_val','y_test']}
    levels={}
    for c in cat:
        vals=arrays['X_train'][:,allf.index(c)]; finite=vals[np.isfinite(vals)]
        lo=min(0,int(np.nanmin(finite))) if finite.size else 0; hi=int(np.nanmax(finite)) if finite.size else 0
        levels[c]=list(range(lo,hi+1))
    def frame(a):
        df=pd.DataFrame(a,columns=allf)
        for c in cat:
            codes=np.rint(df[c].to_numpy()).astype(np.int32)
            df[c]=pd.Series(codes).astype(pd.CategoricalDtype(categories=levels[c],ordered=False))
        for c in cont: df[c]=pd.to_numeric(df[c],errors='coerce').astype(np.float32)
        for c in binary: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0).astype(np.float32)
        return df
    return {**{k:frame(v) for k,v in arrays.items()},**ys,'cat_features':cat,'cont_features':cont,
            'binary_features':binary,'all_features':allf,'category_levels':levels}

def restore_frame(a,d):
    df=pd.DataFrame(np.asarray(a),columns=d['all_features'])
    for c in d['cat_features']:
        allowed=d['category_levels'][c]; codes=np.rint(df[c]).astype(np.int32).clip(min(allowed),max(allowed))
        df[c]=pd.Series(codes).astype(pd.CategoricalDtype(categories=allowed,ordered=False))
    for c in d['cont_features']: df[c]=pd.to_numeric(df[c],errors='coerce').astype(np.float32)
    for c in d['binary_features']: df[c]=np.rint(pd.to_numeric(df[c],errors='coerce').fillna(0)).clip(0,1).astype(np.float32)
    return df

def make_model(extra=None,early=True):
    p=dict(BASE_XGB_PARAMS)
    if not early: p.pop('early_stopping_rounds',None)
    if extra: p.update(extra)
    return xgb.XGBClassifier(**p,tree_method=TREE_METHOD,device=DEVICE,n_jobs=N_JOBS,
                             random_state=RANDOM_SEED,enable_categorical=True,missing=np.nan)

def predict_prob(model,X,batch=BATCH_PREDICTION_ROWS):
    if len(X)<=batch: return model.predict_proba(X)[:,1].astype(np.float64)
    return np.concatenate([model.predict_proba(X.iloc[s:min(s+batch,len(X))])[:,1].astype(np.float64)
                           for s in range(0,len(X),batch)])

def select_threshold(y,p,objective=THRESHOLD_OBJECTIVE,grid=THRESHOLD_GRID_SIZE):
    best_t,best_s=.5,-np.inf
    for t in np.linspace(.001,.999,grid):
        pred=(p>=t).astype(np.int8); s=matthews_corrcoef(y,pred) if objective=='mcc' else f1_score(y,pred,zero_division=0)
        if s>best_s: best_t,best_s=float(t),float(s)
    return best_t,best_s

def metrics(y,p,t):
    pred=(p>=t).astype(np.int8); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(threshold=float(t),accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
        recall_sensitivity=float(recall_score(y,pred,zero_division=0)),specificity=float(tn/(tn+fp)) if tn+fp else np.nan,
        negative_predictive_value=float(tn/(tn+fn)) if tn+fn else np.nan,f1=float(f1_score(y,pred,zero_division=0)),
        roc_auc=float(roc_auc_score(y,p)),pr_auc=float(average_precision_score(y,p)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),mcc=float(matthews_corrcoef(y,pred)),
        cohen_kappa=float(cohen_kappa_score(y,pred)),brier_score=float(brier_score_loss(y,p)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def ratio(y): return float(np.sum(y==0)/np.sum(y==1))

def save_run(method,model,meta,val_prob,test_prob,y_val,y_test):
    md=MODEL_DIR/method; md.mkdir(parents=True,exist_ok=True); model.save_model(md/'model.ubj')
    with (md/'metadata.json').open('w') as f: json.dump(meta,f,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))
    np.savez_compressed(PRED_DIR/f'{method}_predictions.npz',y_val=y_val,val_prob=val_prob.astype(np.float32),y_test=y_test,test_prob=test_prob.astype(np.float32))

def append_result(row):
    p=RESULT_DIR/'all_model_results.csv'; new=pd.DataFrame([row])
    if p.exists():
        old=pd.read_csv(p); old=old[old.method!=row['method']]; new=pd.concat([old,new],ignore_index=True)
    new.to_csv(p,index=False)

def complete(method):
    return (MODEL_DIR/method/'model.ubj').exists() and (MODEL_DIR/method/'metadata.json').exists() and (PRED_DIR/f'{method}_predictions.npz').exists()
