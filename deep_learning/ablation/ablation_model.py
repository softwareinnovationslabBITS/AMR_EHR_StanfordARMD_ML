"""
TabTransformer model classes used by the ablation study.

This architecture mirrors the original training script but supports zero
selected features in any feature family so that feature groups can be fully
removed.

#migrate: extracted from tabtransformer_ablation.py
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return self.norm(x + self.dropout(self.ff(x)))


class AMRTabTransformer(nn.Module):
    """TabTransformer supporting zero selected features in any feature family."""

    def __init__(
        self,
        cat_cardinalities: Dict[str, int],
        cat_embed_dims: Dict[str, int],
        n_cont: int,
        n_bin: int,
        attn_embed_dim: int = 64,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        ff_dim: int = 256,
        mlp_hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_cont = n_cont
        self.n_bin = n_bin

        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Embedding(cardinality, cat_embed_dims[name]),
                    nn.Linear(cat_embed_dims[name], attn_embed_dim),
                )
                for name, cardinality in cat_cardinalities.items()
            ]
        )

        self.transformer_layers = (
            nn.Sequential(
                *[
                    TransformerBlock(
                        attn_embed_dim, num_heads, ff_dim, dropout
                    )
                    for _ in range(num_transformer_layers)
                ]
            )
            if self.n_cat > 0
            else None
        )

        if self.n_cont > 0:
            self.cont_bn = nn.BatchNorm1d(self.n_cont)
            self.cont_proj = nn.Sequential(
                nn.Linear(self.n_cont, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 128),
            )
        else:
            self.cont_bn = None
            self.cont_proj = None

        if self.n_bin > 0:
            self.bin_proj = nn.Sequential(
                nn.Linear(self.n_bin, 64),
                nn.GELU(),
            )
        else:
            self.bin_proj = None

        # Wide branch receives all retained continuous and binary features.
        wide_input_dim = self.n_cont + self.n_bin
        self.wide_proj = (
            nn.Linear(wide_input_dim, 64) if wide_input_dim > 0 else None
        )

        n_cat_out = self.n_cat * attn_embed_dim
        n_cont_out = 128 if self.n_cont > 0 else 0
        n_bin_out = 64 if self.n_bin > 0 else 0
        n_wide_out = 64 if wide_input_dim > 0 else 0
        fused_dim = n_cat_out + n_cont_out + n_bin_out + n_wide_out

        if fused_dim == 0:
            raise ValueError("An ablation cannot remove every available feature.")

        mlp_layers: List[nn.Module] = []
        current_dim = fused_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        mlp_layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        x_bin: torch.Tensor,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []

        if self.n_cat > 0:
            embedded = [
                embedding(x_cat[:, index])
                for index, embedding in enumerate(self.embeddings)
            ]
            cat_sequence = torch.stack(embedded, dim=1)
            cat_sequence = self.transformer_layers(cat_sequence)
            outputs.append(cat_sequence.flatten(1))

        if self.n_cont > 0:
            normalized_cont = self.cont_bn(x_cont)
            outputs.append(self.cont_proj(normalized_cont))
        else:
            normalized_cont = x_cont

        if self.n_bin > 0:
            outputs.append(self.bin_proj(x_bin))

        wide_parts: List[torch.Tensor] = []
        if self.n_cont > 0:
            wide_parts.append(normalized_cont)
        if self.n_bin > 0:
            wide_parts.append(x_bin)
        if wide_parts:
            outputs.append(self.wide_proj(torch.cat(wide_parts, dim=1)))

        fused = torch.cat(outputs, dim=1)
        return self.mlp(fused).squeeze(1)
