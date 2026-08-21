# Source: /AMR_Stanford/DL_codes/amr_DL.py (model-training split)
import logging
import warnings
import gc
import sys
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# #migrate: load training settings from the single config file
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config_loader import load_config, resolve_path

_CFG = load_config()
_TT_CFG = _CFG.get('tabtransformer', {})

BATCH_SIZE = _TT_CFG.get('batch_size', 512)
NUM_EPOCHS = _TT_CFG.get('epochs', 60)
PATIENCE = _TT_CFG.get('patience', 10)
LEARNING_RATE = _TT_CFG.get('learning_rate', 3e-4)
WEIGHT_DECAY = _TT_CFG.get('weight_decay', 1e-4)
MODEL_PATH = resolve_path(_TT_CFG.get('model_path', 'models/tabtransformer/amr_model.pt'))
BUNDLE_PATH = resolve_path(_TT_CFG.get('bundle_path', 'dataset/amr_analysis_bundle.joblib'))

# #migrate: load preprocessing bundle produced by preprocessing/build_dl_features.py
_bundle = joblib.load(BUNDLE_PATH)
CAT_FEATURES = _bundle['CAT_FEATURES']
CONT_FEATURES = _bundle['CONT_FEATURES']
BINARY_FEATURES = _bundle['BINARY_FEATURES']
ALL_FEATURES = _bundle['ALL_FEATURES']
cat_idx = _bundle['cat_idx']
cont_idx = _bundle['cont_idx']
bin_idx = _bundle['bin_idx']
cat_cardinalities = _bundle['cat_cardinalities']
cat_embed_dims = _bundle['cat_embed_dims']
MERGE_KEY = _bundle['MERGE_KEY']
X_train, X_val, X_test = _bundle['X_train'], _bundle['X_val'], _bundle['X_test']
y_train, y_val, y_test = _bundle['y_train'], _bundle['y_val'], _bundle['y_test']
scaler = _bundle['scaler']
org_le = _bundle['org_le']
ab_le = _bundle['ab_le']
keys_train = _bundle['keys_train']
keys_val = _bundle['keys_val']
keys_test = _bundle['keys_test']
logger.info("Loaded bundle -> Train: %d, Val: %d, Test: %d",
            X_train.shape[0], X_val.shape[0], X_test.shape[0])

if __name__ == "__main__":
    logger.info("Starting AMR TabTransformer training")

    device_cfg = _TT_CFG.get('device', 'cuda')
    if device_cfg == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device_cfg)
    logger.info("Using device: %s", device)

    class AMRDataset(Dataset):
        def __init__(self, X, y):
            self.X_cat  = torch.LongTensor(X[:, cat_idx].astype(np.int64))
            self.X_cont = torch.FloatTensor(X[:, cont_idx].astype(np.float32))
            self.X_bin  = torch.from_numpy(X[:, bin_idx].astype(np.uint8))
            self.y      = torch.FloatTensor(y.astype(np.float32))

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return (
                self.X_cat[idx],
                self.X_cont[idx],
                self.X_bin[idx].float(),
                self.y[idx]
            )

    train_dataset = AMRDataset(X_train, y_train)
    val_dataset   = AMRDataset(X_val,   y_val)
    test_dataset  = AMRDataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---------------------------------------------------------
    # 4. Model Architecture
    # ---------------------------------------------------------
    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, dropout=0.1):
            super().__init__()
            self.attn    = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm    = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x)
            return self.norm(x + self.dropout(attn_out))

    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
            super().__init__()
            self.attn        = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
            self.ff          = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim)
            )
            self.norm        = nn.LayerNorm(embed_dim)
            self.dropout     = nn.Dropout(dropout)

        def forward(self, x):
            x = self.attn(x)
            return self.norm(x + self.dropout(self.ff(x)))

    class AMRTabTransformer(nn.Module):
        def __init__(self, cat_cardinalities, cat_embed_dims, n_cont, n_bin,
                     attn_embed_dim=64, num_heads=8, num_transformer_layers=4,
                     ff_dim=256, mlp_hidden_dims=(512, 256, 128), dropout=0.2):
            super().__init__()
            self.cat_feature_names = list(cat_cardinalities.keys())
            self.embeddings = nn.ModuleList([
                nn.Sequential(
                    nn.Embedding(card, cat_embed_dims[feat]),
                    nn.Linear(cat_embed_dims[feat], attn_embed_dim)
                )
                for feat, card in cat_cardinalities.items()
            ])
            self.transformer_layers = nn.Sequential(*[
                TransformerBlock(attn_embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_transformer_layers)
            ])
            self.cont_bn    = nn.BatchNorm1d(n_cont)
            self.cont_proj  = nn.Sequential(
                nn.Linear(n_cont, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 128)
            )
            self.bin_proj = nn.Sequential(
                nn.Linear(n_bin, 64) if n_bin > 0 else nn.Identity(),
                nn.GELU() if n_bin > 0 else nn.Identity()
            )
            self.n_bin = n_bin
            wide_in = n_cont + n_bin
            self.wide_proj = nn.Linear(wide_in, 64)
            n_cat   = len(cat_cardinalities) * attn_embed_dim
            n_deep  = 128 + (64 if n_bin > 0 else 0)
            n_wide  = 64
            total   = n_cat + n_deep + n_wide
            mlp_layers = []
            in_dim = total
            for hidden in mlp_hidden_dims:
                mlp_layers += [
                    nn.Linear(in_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ]
                in_dim = hidden
            mlp_layers += [nn.Linear(in_dim, 1)]
            self.mlp = nn.Sequential(*mlp_layers)
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.01)

        def forward(self, x_cat, x_cont, x_bin):
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            cat_seq    = torch.stack(cat_embeds, dim=1)
            cat_seq    = self.transformer_layers(cat_seq)
            cat_flat   = cat_seq.flatten(1)
            x_cont_bn  = self.cont_bn(x_cont)
            cont_out   = self.cont_proj(x_cont_bn)
            if self.n_bin > 0:
                bin_out = self.bin_proj(x_bin)
                deep_out = torch.cat([cont_out, bin_out], dim=1)
            else:
                deep_out = cont_out
            wide_in  = torch.cat([x_cont_bn, x_bin], dim=1)
            wide_out = self.wide_proj(wide_in)
            fused = torch.cat([cat_flat, deep_out, wide_out], dim=1)
            out   = self.mlp(fused)
            return out.squeeze(1)

    model = AMRTabTransformer(
        cat_cardinalities=cat_cardinalities,
        cat_embed_dims=cat_embed_dims,
        n_cont=len(CONT_FEATURES),
        n_bin=len(BINARY_FEATURES),
    ).to(device)

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    pos_weight_val = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    best_val_loss = float('inf')
    best_val_auc  = 0.0
    patience_cnt  = 0
    best_model_wts = None

    # Per-epoch curves -> saved so the analysis script can plot the real
    # training history instead of re-deriving or guessing it.
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_auc': [], 'val_auc': [],
        'train_pr_auc': [], 'val_pr_auc': []
    }

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for x_cat, x_cont, x_bin, y_batch in train_loader:
            x_cat, x_cont, x_bin, y_batch = x_cat.to(device), x_cont.to(device), x_bin.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_cat, x_cont, x_bin)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        scheduler.step()
        model.eval()
        val_logits_list, val_labels_list = [], []
        total_val_loss = 0

        with torch.no_grad():
            for x_cat, x_cont, x_bin, y_batch in val_loader:
                x_cat, x_cont, x_bin, y_batch_dev = x_cat.to(device), x_cont.to(device), x_bin.to(device), y_batch.to(device)
                logits = model(x_cat, x_cont, x_bin)
                loss   = criterion(logits, y_batch_dev)
                total_val_loss += loss.item() * len(y_batch)
                val_logits_list.append(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.append(y_batch.numpy())

        # Also calculate Training AUC/PR-AUC for diagnostics (takes ~15-20s per epoch)
        train_logits_list, train_labels_list = [], []
        with torch.no_grad():
            for x_cat, x_cont, x_bin, y_batch in train_eval_loader:
                x_cat, x_cont, x_bin = x_cat.to(device), x_cont.to(device), x_bin.to(device)
                logits = model(x_cat, x_cont, x_bin)
                train_logits_list.append(torch.sigmoid(logits).cpu().numpy())
                train_labels_list.append(y_batch.numpy())

        val_probs, val_labels = np.concatenate(val_logits_list), np.concatenate(val_labels_list)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_pr_auc = average_precision_score(val_labels, val_probs)

        train_probs, train_labels = np.concatenate(train_logits_list), np.concatenate(train_labels_list)
        train_auc = roc_auc_score(train_labels, train_probs)
        train_pr_auc = average_precision_score(train_labels, train_probs)

        train_loss_epoch = total_loss / len(train_dataset)
        val_loss_epoch = total_val_loss / len(val_dataset)
        logger.info("Epoch [%02d/%d] train_loss=%.4f val_loss=%.4f train_auc=%.4f val_auc=%.4f train_pr_auc=%.4f val_pr_auc=%.4f",
                    epoch + 1, NUM_EPOCHS, train_loss_epoch, val_loss_epoch, train_auc, val_auc, train_pr_auc, val_pr_auc)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_pr_auc'].append(train_pr_auc)
        history['val_pr_auc'].append(val_pr_auc)

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_val_auc = val_auc
            best_model_wts = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                logger.info("Early stopping triggered at epoch %d due to validation loss not improving", epoch + 1)
                break

    # ---------------------------------------------------------
    # 6. Comprehensive Saving for Post-hoc Analysis
    # ---------------------------------------------------------
    logger.info("Training finished. Best val loss: %.4f. Saving artifacts...", best_val_loss)
    model.load_state_dict(best_model_wts)

    # ---- 6a. Model weights + architecture metadata -> torch.save -----------
    # State dicts (and any GPU tensors) belong in torch.save/torch.load --
    # that's the format PyTorch is designed around, with correct handling
    # of device placement, dtype, and tensor storage layout. joblib would
    # work but offers no benefit here and loses torch's map_location support.
    model_bundle = {
        'model_state_dict':  model.state_dict(),
        'cat_cardinalities': cat_cardinalities,
        'cat_embed_dims':    cat_embed_dims,
        'n_cont':            len(CONT_FEATURES),
        'n_bin':             len(BINARY_FEATURES),
        'best_val_auc':      best_val_auc,
        'history':           history,
    }
    # #migrate: save trained model under models/tabtransformer/
    import os
    os.makedirs(MODEL_PATH.parent, exist_ok=True)
    torch.save(model_bundle, MODEL_PATH)
    logger.info("Saved model weights + metadata -> %s", MODEL_PATH)

    # ---- 6b. Everything else -> joblib --------------------------------------
    # joblib is the standard tool for this: it's pickle-based but with much
    # better compression/throughput for large numpy arrays (X_train/X_val/
    # X_test can easily be hundreds of MB here), and it's the conventional
    # way to persist sklearn objects like StandardScaler / LabelEncoder.
    # compress=3 is a reasonable size/speed tradeoff; raise toward 9 if disk
    # space is the priority, or use 0 for fastest save/load.
    analysis_bundle = {
        # ── feature schema — must match model_bundle's n_cont/n_bin exactly ──
        'CAT_FEATURES':      CAT_FEATURES,
        'CONT_FEATURES':     CONT_FEATURES,
        'BINARY_FEATURES':   BINARY_FEATURES,
        'ALL_FEATURES':      ALL_FEATURES,
        'cat_idx':           cat_idx,
        'cont_idx':          cont_idx,
        'bin_idx':           bin_idx,
        'cat_cardinalities': cat_cardinalities,
        'cat_embed_dims':    cat_embed_dims,
        'MERGE_KEY':         MERGE_KEY,

        # ── preprocessors — needed to transform any new/unseen data identically ──
        'scaler':            scaler,
        'org_le':            org_le,
        'ab_le':             ab_le,

        # ── the exact arrays used for training/eval. Saving these is what
        #    removes ALL dependency on re-running the CSV merge pipeline
        #    identically — the analysis script loads these directly and is
        #    therefore guaranteed to match training-time metrics exactly,
        #    regardless of any future change to the raw CSVs. ──
        'X_train':           X_train,
        'X_val':             X_val,
        'X_test':            X_test,
        'y_train':           y_train,
        'y_val':             y_val,
        'y_test':            y_test,

        # ── merge keys per split, aligned row-for-row with X_*/y_* above.
        #    Useful for tracing any prediction back to order_proc_id_coded
        #    for case-level inspection / error analysis. ──
        'keys_train':        keys_train,
        'keys_val':          keys_val,
        'keys_test':         keys_test,

        # ── misc metadata for sanity-checking consistency across retrains ──
        'n_features_total':  len(ALL_FEATURES),
        'class_weights':     class_weights,
    }
    joblib.dump(analysis_bundle, BUNDLE_PATH, compress=3)
    logger.info("Saved splits + preprocessors + metadata -> %s", BUNDLE_PATH)

    logger.info("SUCCESS: %s  (model weights + history)", MODEL_PATH)
    logger.info("SUCCESS: %s  (splits + preprocessors + feature schema)", BUNDLE_PATH)