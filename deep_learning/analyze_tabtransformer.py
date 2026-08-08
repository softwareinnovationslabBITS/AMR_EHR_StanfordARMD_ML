"""
AMR TabTransformer - Post-hoc Analysis Script (v3)
===================================================
Loads model + analysis artifacts and produces all analysis plots.

Key design:
  - Two-file format (current): amr_model.pt (weights) + amr_analysis_bundle.joblib
    (splits, preprocessors, feature schema). Splits are saved at training time,
    so this script NEVER re-derives the train/val/test arrays — it loads the
    literal arrays used during training, guaranteeing metrics always match.
  - Backward-compat: if only the legacy single-file `amr_model_complete.pt`
    is found, falls back to rebuilding the dataset from raw CSVs.
  - SHAP uses GradientExplainer (BatchNorm+Embedding safe); falls back to
    DeepExplainer(check_additivity=False) if needed.

Outputs: ./amr_analysis_outputs/  (13+ PNG files)
"""

import os, gc, warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import joblib

warnings.filterwarnings('ignore')

# #migrate: load settings from the single config file
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
_TT_CFG = _CFG.get('tabtransformer', {})
_PATHS_CFG = _CFG.get('paths', {})

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH       = resolve_path(_TT_CFG.get('model_path', 'models/tabtransformer/amr_model.pt'))  # weights + history (new format)
ANALYSIS_PATH    = resolve_path(_TT_CFG.get('bundle_path', 'dataset/amr_analysis_bundle.joblib'))  # splits + preprocessors (new format)
LEGACY_PATH      = resolve_path('models/tabtransformer/amr_model_complete.pt')  # old single-file format (fallback)
DATA_DIR         = resolve_path(_PATHS_CFG.get('dataset_dir', 'dataset'))  # change if needed
OUT_DIR          = Path('./amr_analysis_outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3})
PAL = {'R': '#E84A5F', 'S': '#2A9D8F', 'N': '#457B9D'}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must match training script exactly)
# ══════════════════════════════════════════════════════════════════════════════
class MHSA(nn.Module):
    def __init__(self, d, h, dr=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dr, batch_first=True)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(dr)
    def forward(self, x):
        o, _ = self.attn(x, x, x); return self.norm(x + self.drop(o))

class TBlock(nn.Module):
    def __init__(self, d, h, ff, dr=0.1):
        super().__init__()
        self.attn = MHSA(d, h, dr)
        self.ff   = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dr), nn.Linear(ff, d))
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(dr)
    def forward(self, x):
        x = self.attn(x); return self.norm(x + self.drop(self.ff(x)))

class AMRTabTransformer(nn.Module):
    def __init__(self, cat_cardinalities, cat_embed_dims, n_cont, n_bin,
                 attn_embed_dim=64, num_heads=8, num_transformer_layers=4,
                 ff_dim=256, mlp_hidden_dims=(512, 256, 128), dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(nn.Embedding(card, cat_embed_dims[feat]),
                          nn.Linear(cat_embed_dims[feat], attn_embed_dim))
            for feat, card in cat_cardinalities.items()])
        self.transformer_layers = nn.Sequential(*[
            TBlock(attn_embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_layers)])
        self.cont_bn   = nn.BatchNorm1d(n_cont)
        self.cont_proj = nn.Sequential(nn.Linear(n_cont, 128), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(128, 128))
        self.bin_proj  = nn.Sequential(
            nn.Linear(n_bin, 64) if n_bin > 0 else nn.Identity(),
            nn.GELU()            if n_bin > 0 else nn.Identity())
        self.n_bin     = n_bin
        self.wide_proj = nn.Linear(n_cont + n_bin, 64)
        n_cat  = len(cat_cardinalities) * attn_embed_dim
        n_deep = 128 + (64 if n_bin > 0 else 0)
        mlp, d = [], n_cat + n_deep + 64
        for h in mlp_hidden_dims:
            mlp += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        mlp += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x_cat, x_cont, x_bin):
        embs     = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        cat_seq  = self.transformer_layers(torch.stack(embs, dim=1)).flatten(1)
        xc       = self.cont_bn(x_cont)
        co       = self.cont_proj(xc)
        deep     = torch.cat([co, self.bin_proj(x_bin)], 1) if self.n_bin > 0 else co
        wide     = self.wide_proj(torch.cat([xc, x_bin], 1))
        return self.mlp(torch.cat([cat_seq, deep, wide], 1)).squeeze(1)

# ── dataset ────────────────────────────────────────────────────────────────────
class AMRDataset(Dataset):
    def __init__(self, X, y, ci, ni, bi):
        self.Xc = torch.LongTensor(X[:, ci].astype(np.int64))
        self.Xn = torch.FloatTensor(X[:, ni].astype(np.float32))
        self.Xb = torch.FloatTensor(X[:, bi].astype(np.float32))
        self.y  = torch.FloatTensor(y.astype(np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Xc[i], self.Xn[i], self.Xb[i], self.y[i]

@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    ps, ys = [], []
    for xc, xn, xb, yb in loader:
        ps.append(torch.sigmoid(model(xc.to(device), xn.to(device), xb.to(device))).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)

# ── flat wrapper for SHAP ──────────────────────────────────────────────────────
class FlatWrapper(nn.Module):
    """
    Wraps the model for SHAP. Deliberately returns RAW LOGITS, not
    sigmoid-transformed probabilities. SHAP's additivity property
    (sum(shap_values) + base_value == f(x)) only holds for the model's
    direct output -- stacking a nonlinear sigmoid on top breaks it, and
    also crushes gradients into near-zero everywhere a prediction is
    confident (sigmoid saturation), which is why earlier beeswarm/waterfall
    plots showed near-invisible SHAP values and a waterfall that didn't
    add up to the stated probability. Convert logits -> probability only
    at display time, never before SHAP sees the output.
    """
    def __init__(self, base, ci, ni, bi):
        super().__init__()
        self.base = base; self.ci = ci; self.ni = ni; self.bi = bi
    def forward(self, x):
        return self.base(x[:, self.ci].long(), x[:, self.ni], x[:, self.bi]).unsqueeze(1)


# ── legacy rebuild path: only used when no saved splits are found ─────────────
def rebuild_dataset_from_csv(data_dir, all_features, cont_features, cat_features,
                              binary_features, scaler, org_le, ab_le):
    """
    Reconstructs X_train/X_val/X_test by re-reading and re-merging all raw
    CSVs, using the *exact* feature lists and fitted preprocessors saved at
    training time. Only used as a fallback for the legacy single-file bundle
    that didn't save the splits directly -- prefer the new joblib format,
    which loads the literal arrays and never needs this function.
    """
    MERGE_KEY = 'order_proc_id_coded'

    # ── cohort ──
    cohort = pd.read_csv(f'{data_dir}/microbiology_cultures_cohort.csv')
    df = cohort[cohort['susceptibility'].isin(['Susceptible', 'Resistant'])].copy()
    df['label'] = (df['susceptibility'] == 'Resistant').astype(int)
    del cohort; gc.collect()

    df['culture_type_enc'] = df['culture_description'].map(
        {'URINE': 0, 'BLOOD': 1, 'RESPIRATORY': 2}).fillna(-1).astype(int)
    df['ordering_mode_enc'] = df['ordering_mode'].map(
        {'Inpatient': 0, 'Outpatient': 1, 'Null': 2}).fillna(2).astype(int)

    # use saved label encoders -- handles unseen labels gracefully
    known_orgs = set(org_le.classes_)
    df['organism_enc'] = org_le.transform(
        df['organism'].fillna('Unknown').apply(
            lambda x: x if x in known_orgs else 'Unknown'))

    known_abs = set(ab_le.classes_)
    df['antibiotic_enc'] = ab_le.transform(
        df['antibiotic'].fillna('Unknown').apply(
            lambda x: x if x in known_abs else 'Unknown'))

    t = pd.to_datetime(df['order_time_jittered_utc'], utc=True, errors='coerce')
    df['order_year']  = t.dt.year.fillna(2020).astype(int)
    df['order_month'] = t.dt.month.fillna(1).astype(int)

    # ── demographics ──
    age_map = {'18-24 years': 1, '25-34 years': 2, '35-44 years': 3, '45-54 years': 4,
               '55-64 years': 5, '65-74 years': 6, '75-84 years': 7, '85-89 years': 8,
               '90+ years': 9}
    try:
        demo = pd.read_csv(f'{data_dir}/microbiology_cultures_demographics.csv')
        d = demo[[MERGE_KEY, 'age', 'gender']].drop_duplicates(MERGE_KEY).copy()
        d['age_enc']    = d['age'].map(age_map).fillna(5).astype(int)
        d['gender_enc'] = pd.to_numeric(d['gender'], errors='coerce').fillna(0).astype(int)
        df = df.merge(d[[MERGE_KEY, 'age_enc', 'gender_enc']], on=MERGE_KEY, how='left')
        del demo, d; gc.collect()
    except FileNotFoundError:
        print('  [WARN] demographics not found')
    df['age_enc']    = df['age_enc'].fillna(5).astype(int)    if 'age_enc'    in df else 5
    df['gender_enc'] = df['gender_enc'].fillna(0).astype(int) if 'gender_enc' in df else 0

    # ── generic numeric side-table merge ──
    def merge_numeric(df_in, path, cols):
        try:
            s = pd.read_csv(path)
            avail = [c for c in cols if c in s.columns]
            f = s[[MERGE_KEY] + avail].drop_duplicates(MERGE_KEY).copy()
            for c in avail:
                f[c] = pd.to_numeric(f[c].replace('Null', np.nan), errors='coerce')
            df_in = df_in.merge(f, on=MERGE_KEY, how='left')
            for c in avail:
                df_in[c] = df_in[c].fillna(df_in[c].median())
            return df_in
        except FileNotFoundError:
            print(f'  [WARN] {path} not found')
            return df_in

    df = merge_numeric(df, f'{data_dir}/microbiology_cultures_labs.csv',   cont_features)
    df = merge_numeric(df, f'{data_dir}/microbiology_cultures_vitals.csv', cont_features)

    # ── ward ──
    try:
        ward = pd.read_csv(f'{data_dir}/microbiology_cultures_ward_info.csv')
        ward_cols = [c for c in ['hosp_ward_IP', 'hosp_ward_OP', 'hosp_ward_ER', 'hosp_ward_ICU']
                     if c in ward.columns]
        wf = ward[[MERGE_KEY] + ward_cols].drop_duplicates(MERGE_KEY)
        df = df.merge(wf, on=MERGE_KEY, how='left')
        for c in ward_cols:
            df[c] = df[c].fillna(0).astype(int)
        del ward; gc.collect()
    except FileNotFoundError:
        print('  [WARN] ward info not found')

    # ── comorbidities ──
    try:
        cr = pd.read_csv(f'{data_dir}/microbiology_cultures_comorbidity.csv', low_memory=False)
        ca = cr[cr['comorbidity_component_start_days_culture'] >= 0].copy()
        del cr; gc.collect()
        ca['comorb_col'] = ('comorb_' +
            ca['comorbidity_component'].str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_'))
        cc = (ca.groupby(MERGE_KEY)['comorbidity_component'].nunique()
              .reset_index().rename(columns={'comorbidity_component': 'comorb_total_count'}))
        cpiv = pd.get_dummies(ca[[MERGE_KEY, 'comorb_col']].drop_duplicates(),
                              columns=['comorb_col'], prefix='', prefix_sep='')
        cdf  = cpiv.groupby(MERGE_KEY).max().reset_index()
        cdf  = cdf.merge(cc, on=MERGE_KEY, how='left')
        cdf['comorb_total_count'] = cdf['comorb_total_count'].fillna(0).astype(int)
        df   = df.merge(cdf, on=MERGE_KEY, how='left')
        del ca, cpiv, cdf, cc; gc.collect()
    except FileNotFoundError:
        print('  [WARN] comorbidity not found')

    # ── one-hot side tables ──
    def merge_onehot(df_in, path, col, prefix):
        try:
            s = pd.read_csv(path)
            if col not in s.columns:
                return df_in
            piv = pd.get_dummies(s[[MERGE_KEY, col]].drop_duplicates(),
                                 columns=[col], prefix=prefix)
            agg = piv.groupby(MERGE_KEY).max().reset_index()
            df_in = df_in.merge(agg, on=MERGE_KEY, how='left')
            for c in [x for x in agg.columns if x != MERGE_KEY]:
                df_in[c] = df_in[c].fillna(0).astype(int)
            return df_in
        except FileNotFoundError:
            print(f'  [WARN] {path} not found')
            return df_in

    df = merge_onehot(df, f'{data_dir}/microbiology_cultures_antibiotic_class_exposure.csv',
                      'antibiotic_class', 'abclass')
    df = merge_onehot(df, f'{data_dir}/microbiology_cultures_antibiotic_subtype_exposure.csv',
                      'antibiotic_subtype_category', 'absub')
    df = merge_onehot(df, f'{data_dir}/microbiology_culture_prior_infecting_organism.csv',
                      'prior_organism', 'priororg')
    df = merge_onehot(df, f'{data_dir}/microbiology_cultures_priorprocedures.csv',
                      'procedure_description', 'proc')

    # ── nursing home ──
    try:
        nh = pd.read_csv(f'{data_dir}/microbiology_cultures_nursing_home_visits.csv')
        if 'nursing_home_visit_culture' in nh.columns:
            nhf = nh[[MERGE_KEY, 'nursing_home_visit_culture']].groupby(MERGE_KEY).max().reset_index()
            nhf.rename(columns={'nursing_home_visit_culture': 'nursing_home_visits'}, inplace=True)
            df = df.merge(nhf, on=MERGE_KEY, how='left')
        del nh; gc.collect()
    except FileNotFoundError:
        print('  [WARN] nursing home not found')

    # ── ADI ──
    try:
        adi = pd.read_csv(f'{data_dir}/microbiology_cultures_adi_scores.csv')
        if 'adi_score' in adi.columns:
            af = adi[[MERGE_KEY, 'adi_score', 'adi_state_rank']].drop_duplicates(MERGE_KEY).copy()
            for c in ['adi_score', 'adi_state_rank']:
                af[c] = pd.to_numeric(af[c].replace('Null', np.nan), errors='coerce')
            df = df.merge(af, on=MERGE_KEY, how='left')
        del adi; gc.collect()
    except FileNotFoundError:
        print('  [WARN] ADI not found')

    # ── microbial resistance history ──
    try:
        mr = pd.read_csv(f'{data_dir}/microbiology_cultures_microbial_resistance.csv')
        if 'resistant_time_to_culturetime' in mr.columns:
            mra = (mr[[MERGE_KEY, 'resistant_time_to_culturetime']]
                   .groupby(MERGE_KEY)['resistant_time_to_culturetime']
                   .agg(['count', 'min']).reset_index())
            mra.columns = [MERGE_KEY, 'prior_resistance_count', 'min_resistance_days']
            df = df.merge(mra, on=MERGE_KEY, how='left')
        del mr; gc.collect()
    except FileNotFoundError:
        print('  [WARN] microbial resistance not found')

    # ── guarantee every feature column exists, then fill NaNs ──
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    df_model = df[all_features + ['label']].copy()

    # impute: cont -> median, cat -> 0, binary -> 0
    for col in cont_features:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
        df_model[col] = df_model[col].fillna(df_model[col].median())
    for col in cat_features:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0).astype(int)
    for col in binary_features:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0).astype(int)

    # scale continuous features using the saved scaler (no refit)
    df_model[cont_features] = scaler.transform(df_model[cont_features])

    X = df_model[all_features].values.astype(np.float32)
    y = df_model['label'].values.astype(np.float32)
    del df, df_model; gc.collect()

    # reproduce exact same splits as training
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15, stratify=y_tv, random_state=42)
    del X, y, X_tv, y_tv; gc.collect()

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    device_cfg = _TT_CFG.get('device', 'cuda')
    if device_cfg == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device_cfg)
    print(f'Device: {device}')
    B = _TT_CFG.get('batch_size', 512)

    # ── 1. Load artifacts ──────────────────────────────────────────────────────
    #   PREFERRED: new two-file format — splits already saved, no CSV rebuild
    #   FALLBACK : legacy single-file .pt — rebuild dataset from CSVs
    # ──────────────────────────────────────────────────────────────────────────
    use_new_format = os.path.exists(MODEL_PATH) and os.path.exists(ANALYSIS_PATH)

    if use_new_format:
        print(f'Loading {MODEL_PATH} + {ANALYSIS_PATH} (new format)...')
        model_bundle    = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        analysis_bundle = joblib.load(ANALYSIS_PATH)

        cat_cardinalities = model_bundle['cat_cardinalities']
        cat_embed_dims    = model_bundle['cat_embed_dims']
        n_cont            = model_bundle['n_cont']
        n_bin             = model_bundle['n_bin']
        training_history  = model_bundle.get('history', None)

        CAT_FEATURES      = analysis_bundle['CAT_FEATURES']
        CONT_FEATURES     = analysis_bundle['CONT_FEATURES']
        BINARY_FEATURES   = analysis_bundle['BINARY_FEATURES']
        ALL_FEATURES      = analysis_bundle['ALL_FEATURES']
        scaler            = analysis_bundle['scaler']
        org_le            = analysis_bundle['org_le']
        ab_le             = analysis_bundle['ab_le']

        model = AMRTabTransformer(cat_cardinalities, cat_embed_dims, n_cont, n_bin).to(device)
        model.load_state_dict(model_bundle['model_state_dict'])
        model.eval()
        print('Model loaded.')

        cat_idx  = list(range(len(CAT_FEATURES)))
        cont_idx = list(range(len(CAT_FEATURES), len(CAT_FEATURES) + len(CONT_FEATURES)))
        bin_idx  = list(range(len(CAT_FEATURES) + len(CONT_FEATURES), len(ALL_FEATURES)))

        # ── splits: loaded directly, no rebuild, no drift risk ──
        print('Loading splits from joblib bundle (no CSV rebuild needed)...')
        X_train = analysis_bundle['X_train'].astype(np.float32)
        X_val   = analysis_bundle['X_val'].astype(np.float32)
        X_test  = analysis_bundle['X_test'].astype(np.float32)
        y_train = analysis_bundle['y_train'].astype(np.float32)
        y_val   = analysis_bundle['y_val'].astype(np.float32)
        y_test  = analysis_bundle['y_test'].astype(np.float32)

    else:
        print(f'New-format files not found — falling back to legacy {LEGACY_PATH} ...')
        bundle = torch.load(LEGACY_PATH, map_location='cpu', weights_only=False)
        training_history = None  # legacy bundle never saved per-epoch history

        cat_cardinalities = bundle['cat_cardinalities']
        cat_embed_dims    = bundle['cat_embed_dims']
        CAT_FEATURES      = bundle['CAT_FEATURES']
        CONT_FEATURES     = bundle['CONT_FEATURES']
        BINARY_FEATURES   = bundle['BINARY_FEATURES']
        ALL_FEATURES      = bundle['ALL_FEATURES']
        scaler            = bundle['scaler']
        org_le            = bundle['org_le']
        ab_le             = bundle['ab_le']
        n_cont            = bundle['n_cont']
        n_bin             = bundle['n_bin']

        model = AMRTabTransformer(cat_cardinalities, cat_embed_dims, n_cont, n_bin).to(device)
        model.load_state_dict(bundle['model_state_dict'])
        model.eval()
        print('Model loaded.')

        cat_idx  = list(range(len(CAT_FEATURES)))
        cont_idx = list(range(len(CAT_FEATURES), len(CAT_FEATURES) + len(CONT_FEATURES)))
        bin_idx  = list(range(len(CAT_FEATURES) + len(CONT_FEATURES), len(ALL_FEATURES)))

        if 'X_test' in bundle:
            print('Loading splits from legacy bundle (fast path)...')
            X_train = bundle['X_train'].astype(np.float32)
            X_val   = bundle['X_val'].astype(np.float32)
            X_test  = bundle['X_test'].astype(np.float32)
            y_train = bundle['y_train'].astype(np.float32)
            y_val   = bundle['y_val'].astype(np.float32)
            y_test  = bundle['y_test'].astype(np.float32)
        else:
            print('Splits not in legacy bundle -- rebuilding from CSVs...')
            X_train, X_val, X_test, y_train, y_val, y_test = rebuild_dataset_from_csv(
                DATA_DIR, ALL_FEATURES, CONT_FEATURES, CAT_FEATURES,
                BINARY_FEATURES, scaler, org_le, ab_le
            )

    print(f'Splits → Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}')
    print(f'  [diagnostic] X_test shape: {X_test.shape} | y_test positive rate: {y_test.mean():.4f}')
    print(f'  [diagnostic] Bundle expects {len(ALL_FEATURES)} features '
          f'(cont={len(CONT_FEATURES)}, bin={len(BINARY_FEATURES)}, cat={len(CAT_FEATURES)})')

    # ── 3. DataLoaders + predictions ──────────────────────────────────────────
    # #migrate: batch size from config (defaults to 512 to match training)
    B = _TT_CFG.get('batch_size', 512)
    test_loader  = DataLoader(AMRDataset(X_test,  y_test,  cat_idx, cont_idx, bin_idx), batch_size=B)
    val_loader   = DataLoader(AMRDataset(X_val,   y_val,   cat_idx, cont_idx, bin_idx), batch_size=B)
    train_loader = DataLoader(AMRDataset(X_train, y_train, cat_idx, cont_idx, bin_idx), batch_size=B)

    test_probs,  test_labels  = get_probs(model, test_loader,  device)
    val_probs,   val_labels   = get_probs(model, val_loader,   device)
    train_probs, train_labels = get_probs(model, train_loader, device)

    test_auc   = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)
    val_auc    = roc_auc_score(val_labels,  val_probs)
    train_auc  = roc_auc_score(train_labels, train_probs)
    print(f'\nTrain AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f}')

    thresholds  = np.linspace(0.01, 0.99, 200)
    f1s         = [f1_score(test_labels, (test_probs >= t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1s)]
    test_preds  = (test_probs >= best_thresh).astype(int)
    print(f'Best F1 threshold: {best_thresh:.3f}  (F1={max(f1s):.4f})')

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 0 — Training curves (only available with new-format bundle)
    # ══════════════════════════════════════════════════════════════════════════
    if training_history is not None and len(training_history.get('train_loss', [])) > 0:
        print('[0/9] Training curves...')
        epochs = range(1, len(training_history['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.plot(epochs, training_history['train_loss'], color=PAL['N'], lw=2)
        ax.set(xlabel='Epoch', ylabel='Train Loss', title='Training Loss per Epoch')

        ax = axes[1]
        ax.plot(epochs, training_history['val_auc'], color=PAL['R'], lw=2)
        best_ep = int(np.argmax(training_history['val_auc'])) + 1
        ax.axvline(best_ep, ls='--', color='black', lw=1,
                   label=f'Best epoch = {best_ep} (AUC={max(training_history["val_auc"]):.4f})')
        ax.set(xlabel='Epoch', ylabel='Validation AUC', title='Validation AUC per Epoch')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/00_training_curves.png', bbox_inches='tight')
        plt.close()
    else:
        print('[0/9] Training curves skipped (no history in legacy bundle).')

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1 — ROC & PR
    # ══════════════════════════════════════════════════════════════════════════
    print('\n[1/9] ROC & PR curves...')
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    prec, rec, _ = precision_recall_curve(test_labels, test_probs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(fpr, tpr, color=PAL['R'], lw=2, label=f'AUC = {test_auc:.4f}')
    ax.plot([0,1],[0,1],'k--',lw=1); ax.fill_between(fpr, tpr, alpha=0.1, color=PAL['R'])
    ax.set(xlabel='FPR', ylabel='TPR', title='ROC Curve (Test Set)'); ax.legend(loc='lower right')

    ax = axes[1]
    ax.plot(rec, prec, color=PAL['S'], lw=2, label=f'AUPRC = {test_auprc:.4f}')
    ax.axhline(test_labels.mean(), ls='--', color='gray', lw=1,
               label=f'Baseline = {test_labels.mean():.3f}')
    ax.fill_between(rec, prec, alpha=0.1, color=PAL['S'])
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve (Test Set)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/01_roc_pr_curves.png', bbox_inches='tight'); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2 — Confusion Matrix
    # ══════════════════════════════════════════════════════════════════════════
    print('[2/9] Confusion matrix...')
    cm      = confusion_matrix(test_labels, test_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title, fmt in zip(axes,
                                    [cm, cm_norm],
                                    ['Counts', 'Row-Normalised'],
                                    ['d', '.2f']):
        sns.heatmap(mat, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['Susceptible','Resistant'],
                    yticklabels=['Susceptible','Resistant'],
                    ax=ax, linewidths=0.5)
        ax.set(xlabel='Predicted', ylabel='True',
               title=f'Confusion Matrix ({title})\nThreshold = {best_thresh:.3f}')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/02_confusion_matrix.png', bbox_inches='tight'); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 3 — Threshold analysis
    # ══════════════════════════════════════════════════════════════════════════
    print('[3/9] Threshold analysis...')
    precs = [precision_score(test_labels, (test_probs>=t).astype(int), zero_division=0) for t in thresholds]
    recs_ = [recall_score(test_labels,    (test_probs>=t).astype(int), zero_division=0) for t in thresholds]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1s,   lw=2, color=PAL['R'],  label='F1')
    ax.plot(thresholds, precs, lw=2, color=PAL['N'],  label='Precision')
    ax.plot(thresholds, recs_, lw=2, color=PAL['S'],  label='Recall')
    ax.axvline(best_thresh, ls='--', color='black', lw=1.5,
               label=f'Best F1 = {best_thresh:.3f}')
    ax.set(xlabel='Threshold', ylabel='Score',
           title='Precision / Recall / F1 vs Threshold', xlim=(0,1), ylim=(0,1))
    ax.legend(); plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/03_threshold_analysis.png', bbox_inches='tight'); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 4 — Score distributions
    # ══════════════════════════════════════════════════════════════════════════
    print('[4/9] Score distributions...')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (probs, labels, split) in zip(axes, [
            (test_probs,  test_labels,  'Test'),
            (train_probs, train_labels, 'Train')]):
        ax.hist(probs[labels==0], bins=50, alpha=0.65, color=PAL['S'], density=True, label='Susceptible')
        ax.hist(probs[labels==1], bins=50, alpha=0.65, color=PAL['R'], density=True, label='Resistant')
        ax.axvline(best_thresh, ls='--', color='black', lw=1.5, label=f'Thr={best_thresh:.3f}')
        ax.set(xlabel='Predicted Probability', ylabel='Density', title=f'Score Distribution ({split})')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/04_score_distributions.png', bbox_inches='tight'); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 5 — Calibration
    # ══════════════════════════════════════════════════════════════════════════
    print('[5/9] Calibration curve...')
    fp, mp = calibration_curve(test_labels, test_probs, n_bins=15)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],'k--',lw=1,label='Perfect')
    ax.plot(mp, fp, 'o-', color=PAL['R'], lw=2, label='Model')
    ax.fill_between(mp, fp, mp, alpha=0.15, color=PAL['R'])
    ax.set(xlabel='Mean Predicted Prob', ylabel='Fraction of Positives',
           title='Calibration Curve (Test Set)', xlim=(0,1), ylim=(0,1))
    ax.legend(); plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/05_calibration_curve.png', bbox_inches='tight'); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 6 — Per-class metrics heatmap
    # ══════════════════════════════════════════════════════════════════════════
    print('[6/9] Per-class metrics...')
    rep = classification_report(test_labels, test_preds,
                                target_names=['Susceptible','Resistant'], output_dict=True)
    mdf = (pd.DataFrame(rep).T
           .drop(['accuracy','macro avg','weighted avg'], errors='ignore')
           [['precision','recall','f1-score']].astype(float))
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.heatmap(mdf, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5,
                ax=ax, vmin=0, vmax=1)
    ax.set(title='Per-Class Metrics (Test Set)', ylabel='')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/06_per_class_metrics.png', bbox_inches='tight'); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 7 — SHAP
    # ══════════════════════════════════════════════════════════════════════════
    print('[7/9] SHAP values (may take a few minutes)...')
    N_BG, N_EVAL = 200, 500

    flat = FlatWrapper(model, cat_idx, cont_idx, bin_idx).to(device)
    flat.eval()

    rng     = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_train), N_BG,   replace=False)
    ev_idx  = rng.choice(len(X_test),  N_EVAL, replace=False)
    X_bg    = torch.FloatTensor(X_train[bg_idx]).to(device)
    X_ev    = torch.FloatTensor(X_test[ev_idx]).to(device)
    X_ev_np = X_ev.cpu().numpy()

    # helper functions defined BEFORE use
    def run_gradient(fm, bg, ev):
        exp  = shap.GradientExplainer(fm, bg)
        vals = exp.shap_values(ev)
        if isinstance(vals, list): vals = vals[0]
        if vals.ndim == 3: vals = vals[:,:,0]
        return vals

    def run_deep_nocheck(fm, bg, ev):
        exp  = shap.DeepExplainer(fm, bg)
        vals = exp.shap_values(ev, check_additivity=False)
        if isinstance(vals, list): vals = vals[0]
        if vals.ndim == 3: vals = vals[:,:,0]
        return vals

    sv = None
    for name, fn in [('GradientExplainer', run_gradient),
                     ('DeepExplainer(no_check)', run_deep_nocheck)]:
        try:
            print(f'  Trying {name}...')
            sv = fn(flat, X_bg, X_ev)
            print(f'  {name} succeeded.')
            break
        except Exception as e:
            print(f'  {name} failed: {e}')

    if sv is None:
        print('  All SHAP strategies failed — skipping SHAP plots.')
    else:
        feat_names    = ALL_FEATURES
        mean_abs      = np.abs(sv).mean(axis=0)
        top30_idx     = np.argsort(mean_abs)[::-1][:30]
        top20_idx     = top30_idx[:20]

        # 7a — bar importance
        fig, ax = plt.subplots(figsize=(10, 9))
        vals30  = mean_abs[top30_idx]
        colors  = [PAL['R'] if v > np.median(vals30) else PAL['S'] for v in vals30]
        ax.barh(range(30), vals30[::-1], color=colors[::-1])
        ax.set_yticks(range(30))
        ax.set_yticklabels([feat_names[i] for i in top30_idx][::-1], fontsize=8)
        ax.set(xlabel='Mean |SHAP| (logit scale)',
               title='Top 30 Features — Global Importance (logit space)')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/07a_shap_bar_importance.png', bbox_inches='tight'); plt.close()

        # 7b — beeswarm
        # Figure sized wider to accommodate the SHAP colorbar + long feature
        # names without clipping the x-axis label. SHAP values are now in
        # logit space, so the scale should be meaningfully wider than the
        # near-zero range seen with sigmoid-wrapped output.
        fig = plt.figure(figsize=(14, 9))
        shap.summary_plot(sv[:, top20_idx], X_ev_np[:, top20_idx],
                          feature_names=[feat_names[i] for i in top20_idx],
                          show=False, plot_type='dot', max_display=20)
        fig = plt.gcf()
        fig.set_size_inches(14, 9)
        plt.title('SHAP Beeswarm (logit space) — Top 20 Features', fontsize=13, pad=15)
        plt.xlabel('SHAP value (impact on model output, log-odds scale)', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/07b_shap_beeswarm.png', bbox_inches='tight', dpi=150)
        plt.close(); plt.clf()

        # 7c — waterfall for highest-risk sample
        # base_val is now in LOGIT space (matches sv, which also explains
        # logits). We convert to probability only in the title for display.
        with torch.no_grad():
            base_val_logit = float(flat(X_bg).cpu().numpy().mean())

        hi = np.argmax(test_probs[ev_idx])
        exp_obj = shap.Explanation(
            values=sv[hi], base_values=base_val_logit,
            data=X_ev_np[hi], feature_names=feat_names)

        # sanity check: base_value + sum(shap) should reconstruct the logit
        # (and therefore the probability) for this sample
        reconstructed_logit = base_val_logit + sv[hi].sum()
        reconstructed_prob  = 1 / (1 + np.exp(-reconstructed_logit))
        actual_prob         = test_probs[ev_idx[hi]]
        print(f'  [SHAP sanity check] reconstructed P={reconstructed_prob:.4f} '
              f'vs actual P={actual_prob:.4f} '
              f'(diff={abs(reconstructed_prob - actual_prob):.4f})')

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(exp_obj, max_display=20, show=False)
        plt.title(f'SHAP Waterfall (logit space) — Highest-Risk Sample\n'
                  f'P={actual_prob:.3f}  (reconstructed from SHAP: {reconstructed_prob:.3f})')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/07c_shap_waterfall_high_risk.png', bbox_inches='tight')
        plt.close(); plt.clf()

        # 7d — dependence plots (top 4)
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for ax, fi in zip(axes.flat, top20_idx[:4]):
            ax.scatter(X_ev_np[:, fi], sv[:, fi],
                       c=sv[:, fi], cmap='RdYlGn_r', alpha=0.5, s=15)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set(xlabel=feat_names[fi], ylabel='SHAP (logit scale)',
                   title=f'Dependence: {feat_names[fi]}')
        plt.suptitle('SHAP Dependence (logit space) — Top 4 Features', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/07d_shap_dependence.png', bbox_inches='tight'); plt.close()

    del flat, X_bg, X_ev; gc.collect()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 8 — Embedding t-SNE
    # ══════════════════════════════════════════════════════════════════════════
    print('[8/9] Embedding t-SNE...')
    for feat_idx, feat_name, le_obj in [
        (CAT_FEATURES.index('organism_enc'),   'Organism',   org_le),
        (CAT_FEATURES.index('antibiotic_enc'), 'Antibiotic', ab_le),
    ]:
        W = model.embeddings[feat_idx][0].weight.detach().cpu().numpy()
        n = W.shape[0]
        if n < 4: continue
        perp   = min(30, n - 1)
        coords = TSNE(n_components=2, perplexity=perp, random_state=42,
                      max_iter=1000).fit_transform(W)
        labels = list(le_obj.classes_)[:n]

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(coords[:,0], coords[:,1], c=np.arange(n),
                   cmap='tab20', alpha=0.8,
                   s=np.clip(np.log1p(np.ones(n)*10)*15, 20, 400))
        for i, lbl in enumerate(labels[:40]):
            ax.annotate(lbl, (coords[i,0], coords[i,1]), fontsize=5, alpha=0.75)
        ax.set(title=f't-SNE of {feat_name} Embeddings',
               xlabel='t-SNE 1', ylabel='t-SNE 2')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/08_embedding_tsne_{feat_name.lower()}.png',
                    bbox_inches='tight'); plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 9 — Summary dashboard
    # ══════════════════════════════════════════════════════════════════════════
    print('[9/9] Summary dashboard...')
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(fpr, tpr, color=PAL['R'], lw=2, label=f'AUC={test_auc:.3f}')
    ax.plot([0,1],[0,1],'k--',lw=1); ax.set(title='ROC', xlabel='FPR', ylabel='TPR')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0,1])
    ax.plot(rec, prec, color=PAL['S'], lw=2, label=f'AUPRC={test_auprc:.3f}')
    ax.axhline(test_labels.mean(), ls='--', color='gray', lw=1)
    ax.set(title='Precision-Recall', xlabel='Recall', ylabel='Precision')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0,2])
    ax.plot([0,1],[0,1],'k--',lw=1); ax.plot(mp, fp,'o-',color=PAL['R'],lw=2)
    ax.set(title='Calibration', xlabel='Mean Pred', ylabel='Frac Pos',
           xlim=(0,1), ylim=(0,1))

    ax = fig.add_subplot(gs[1,0])
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['S','R'], yticklabels=['S','R'],
                ax=ax, linewidths=0.5, cbar=False)
    ax.set(title='Confusion (norm)', xlabel='Pred', ylabel='True')

    ax = fig.add_subplot(gs[1,1])
    ax.hist(test_probs[test_labels==0], bins=40, alpha=0.65, color=PAL['S'],
            density=True, label='S')
    ax.hist(test_probs[test_labels==1], bins=40, alpha=0.65, color=PAL['R'],
            density=True, label='R')
    ax.axvline(best_thresh, ls='--', color='k', lw=1)
    ax.set(title='Score Distribution', xlabel='Prob'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1,2])
    if sv is not None:
        top10 = np.argsort(mean_abs)[::-1][:10]
        ax.barh(range(10), mean_abs[top10][::-1], color=PAL['R'])
        ax.set_yticks(range(10))
        ax.set_yticklabels([feat_names[i] for i in top10][::-1], fontsize=7)
        ax.set(title='Top-10 SHAP Features', xlabel='Mean |SHAP|')
    else:
        ax.text(0.5, 0.5, 'SHAP unavailable', ha='center', va='center',
                transform=ax.transAxes)

    fig.suptitle(
        f'AMR TabTransformer  |  AUC={test_auc:.4f}  AUPRC={test_auprc:.4f}  '
        f'F1={max(f1s):.4f}  Thr={best_thresh:.3f}',
        fontsize=12, fontweight='bold')
    plt.savefig(f'{OUT_DIR}/09_summary_dashboard.png', bbox_inches='tight', dpi=180)
    plt.close()

    # ── done ───────────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('OUTPUTS:', os.path.abspath(OUT_DIR))
    print('='*60)
    for f in sorted(os.listdir(OUT_DIR)):
        print(f'  {f}')
    print(f'\n  Train AUC : {train_auc:.4f}')
    print(f'  Val   AUC : {val_auc:.4f}')
    print(f'  Test  AUC : {test_auc:.4f}')
    print(f'  AUPRC     : {test_auprc:.4f}')
    print(f'  Best F1   : {max(f1s):.4f}  @ thr={best_thresh:.3f}')
    print('='*60)
