import warnings
import gc
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

# #migrate: load preprocessing bundle produced by preprocessing/build_dl_features.py
_bundle = joblib.load('../dataset/amr_analysis_bundle.joblib')
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
print(f"Loaded bundle -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

if __name__ == "__main__":
    print("=== Starting AMR Tabular Transformer Training ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

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

    BATCH_SIZE = 512
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    NUM_EPOCHS    = 60
    PATIENCE      = 10
    best_val_auc  = 0.0
    patience_cnt  = 0
    best_model_wts = None

    # Per-epoch curves -> saved so the analysis script can plot the real
    # training history instead of re-deriving or guessing it.
    history = {'train_loss': [], 'val_auc': []}

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

        with torch.no_grad():
            for x_cat, x_cont, x_bin, y_batch in val_loader:
                x_cat, x_cont, x_bin = x_cat.to(device), x_cont.to(device), x_bin.to(device)
                logits = model(x_cat, x_cont, x_bin)
                val_logits_list.append(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.append(y_batch.numpy())

        val_probs, val_labels = np.concatenate(val_logits_list), np.concatenate(val_labels_list)
        val_auc = roc_auc_score(val_labels, val_probs)
        train_loss_epoch = total_loss / len(train_dataset)
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] - Train Loss: {train_loss_epoch:.4f} | Val AUC: {val_auc:.4f}")

        history['train_loss'].append(train_loss_epoch)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping triggered.")
                break

    # ---------------------------------------------------------
    # 6. Comprehensive Saving for Post-hoc Analysis
    # ---------------------------------------------------------
    print("\nTraining Finished. Packing up model and processing artifacts...")
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
    os.makedirs('../models/tabtransformer', exist_ok=True)
    torch.save(model_bundle, '../models/tabtransformer/amr_model.pt')
    print("Saved model weights + architecture metadata -> ../models/tabtransformer/amr_model.pt")

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
    joblib.dump(analysis_bundle, 'amr_analysis_bundle.joblib', compress=3)
    print("Saved splits + preprocessors + metadata -> amr_analysis_bundle.joblib")

    print("\nSUCCESS: training artifacts saved as:")
    print("  - ../models/tabtransformer/amr_model.pt  (model weights + history)")
    print("  - ../dataset/amr_analysis_bundle.joblib  (splits + preprocessors + feature schema)")
