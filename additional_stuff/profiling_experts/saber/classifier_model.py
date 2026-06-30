"""SABER token -> cluster classifier (paper section 5.1.3).

Lightweight architecture:
  - learned token embedding over token_id
  - one sink+window self-attention layer over the selected context tokens
    (n_sink sequence-start "sink" tokens + n_recent most-recent tokens + target)
  - 2-layer MLP head on the target token's representation -> softmax over |C*|

The sink+window token selection is done in the dataset builder; this module just
embeds the (already selected) token sequence and attends over it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SinkWindowClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_clusters: int,
        d_model: int = 128,
        n_heads: int = 4,
        seq_len: int = 33,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.attn = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_clusters),
        )
        self.seq_len = seq_len
        self.num_clusters = num_clusters

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, S) int. Target token is the last position. Returns logits (B, C)."""
        pad_mask = tokens == 0  # (B, S) True where padding
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1], :]
        h = self.attn(x, src_key_padding_mask=pad_mask)  # (B, S, D)
        target = h[:, -1, :]  # representation of the target token
        return self.head(target)
