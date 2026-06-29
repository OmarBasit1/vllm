"""Datasets for the past-token and past-layer expert-prediction experiments.

Both read the shared compact store produced by `seq_dataset_builder.py` and build, per
sample, a feature made of per-item encodings:

    item(token', layer') = [K sorted top-k prob values] ++ [K rank-ordered one-hot(E)]
                          = K + K*E  dims   (default 4 + 240 = 244)

Targets are the full E-dim softmax at (token t, layer i). KL loss is applied downstream.

PastTokenDataset(layer_i, W): history of the same layer i over the previous W decode
tokens. Feature dim = W*(K+K*E) + W (presence mask).

PastLayerDataset(layer_i, N): the previous N (token, layer) entries in global decode order
(wrapping across token/iteration boundaries). Feature dim = N*(K+K*E) + N (mask)
+ L (multi-hot of which absolute layers are present in the window).

History that would reach before a request's first token is zero-padded; a presence mask
marks padded slots. Windows never cross request boundaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")


def _load_store(data_dir: Path, split: str):
    """Return (targets_mmap, topk_idx, topk_val, req_starts) or None if split missing."""
    tgt_path = data_dir / f"seq_{split}_targets.npy"
    aux_path = data_dir / f"seq_{split}_aux.npz"
    if not tgt_path.exists() or not aux_path.exists():
        return None
    targets = np.load(tgt_path, mmap_mode="r")          # (Ntok, L, E) — lazy
    aux = np.load(aux_path)
    return targets, aux["topk_idx"], aux["topk_val"], aux["req_starts"]


def _req_start_per_token(req_starts: np.ndarray, n_tok: int) -> np.ndarray:
    """Map each token index → the start-token of its request."""
    lengths = np.diff(req_starts)
    return np.repeat(req_starts[:-1], lengths).astype(np.int64)


class _SeqDatasetBase(Dataset):
    """Shared machinery: holds compact top-k arrays + precomputed gather plan."""

    def __init__(
        self,
        layer_idx: int,
        window: int,
        split: str,
        data_dir: str | Path,
        missing_ok: bool = False,
    ) -> None:
        assert split in SPLITS, f"split must be one of {SPLITS}"
        data_dir = Path(data_dir)
        self.layer_idx = layer_idx
        self.window = window
        self.split = split

        store = _load_store(data_dir, split)
        if store is None:
            if missing_ok:
                self._init_empty()
                return
            raise FileNotFoundError(f"missing store for split '{split}' in {data_dir}")

        targets, topk_idx, topk_val, req_starts = store
        self.n_tok, self.L, self.K = topk_idx.shape
        self.E = targets.shape[2]
        assert layer_idx < self.L, f"layer {layer_idx} >= num_layers {self.L}"

        # Targets for this layer only (cheap mmap slice → small copy).
        self.y = torch.from_numpy(np.array(targets[:, layer_idx, :], dtype=np.float32))

        # Flatten (token, layer) → item axis for global indexing.
        self.idx_flat = torch.from_numpy(
            topk_idx.reshape(self.n_tok * self.L, self.K).astype(np.int64)
        )
        self.val_flat = torch.from_numpy(
            topk_val.reshape(self.n_tok * self.L, self.K).astype(np.float32)
        )

        rspt = _req_start_per_token(req_starts, self.n_tok)  # (Ntok,)
        self._build_plan(rspt)  # subclass fills self.gather, self.mask, self.extra
        self.n_samples = self.n_tok

    def _init_empty(self) -> None:
        self.n_tok = self.n_samples = 0
        self.L, self.K, self.E = 24, 4, 60
        self.y = torch.empty(0, self.E)
        self.idx_flat = torch.empty(0, self.K, dtype=torch.long)
        self.val_flat = torch.empty(0, self.K)
        self.gather = torch.empty(0, self.window, dtype=torch.long)
        self.mask = torch.empty(0, self.window)
        self.extra = torch.empty(0, 0)
        self.input_dim = self._input_dim()

    # ── subclass hooks ──
    def _build_plan(self, rspt: np.ndarray) -> None:
        raise NotImplementedError

    def _input_dim(self) -> int:
        raise NotImplementedError

    # ── common ──
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, t: int):
        gather = self.gather[t]          # (S,) flat item indices (clamped)
        mask = self.mask[t]              # (S,) 1=real, 0=pad
        vals = self.val_flat[gather]     # (S, K)
        idxs = self.idx_flat[gather]     # (S, K)
        S = gather.shape[0]
        onehot = torch.zeros(S, self.K, self.E)
        onehot.scatter_(2, idxs.unsqueeze(-1), 1.0)  # rank-ordered one-hots
        m = mask.view(S, 1)
        per_slot = torch.cat([vals * m, (onehot * m.unsqueeze(-1)).reshape(S, self.K * self.E)], dim=1)
        feat = per_slot.reshape(-1)                  # (S*(K+K*E),)
        parts = [feat, mask]
        if self.extra.numel() > 0:
            parts.append(self.extra[t])
        x = torch.cat(parts)
        return x, self.y[t]


class PastTokenDataset(_SeqDatasetBase):
    """Same-layer history over the previous W decode tokens."""

    def _input_dim(self) -> int:
        return self.window * (self.K + self.K * self.E) + self.window

    def _build_plan(self, rspt: np.ndarray) -> None:
        W, L, i = self.window, self.L, self.layer_idx
        t = np.arange(self.n_tok)
        gather = np.zeros((self.n_tok, W), dtype=np.int64)
        mask = np.zeros((self.n_tok, W), dtype=np.float32)
        for w in range(1, W + 1):
            src_tok = t - w
            valid = src_tok >= rspt          # within same request
            flat = src_tok * L + i
            gather[:, w - 1] = np.where(valid, flat, 0)
            mask[:, w - 1] = valid.astype(np.float32)
        self.gather = torch.from_numpy(gather)
        self.mask = torch.from_numpy(mask)
        self.extra = torch.empty(self.n_tok, 0)
        self.input_dim = self._input_dim()


class PastLayerDataset(_SeqDatasetBase):
    """Previous N (token, layer) entries in global decode order (wraps across tokens),
    plus an L-dim multi-hot of which absolute layers are present in the window."""

    def _input_dim(self) -> int:
        return self.window * (self.K + self.K * self.E) + self.window + 24

    def _build_plan(self, rspt: np.ndarray) -> None:
        N, L, i = self.window, self.L, self.layer_idx
        t = np.arange(self.n_tok)
        g = t * L + i                        # current global index
        req_start_global = rspt * L          # first global index of the request
        gather = np.zeros((self.n_tok, N), dtype=np.int64)
        mask = np.zeros((self.n_tok, N), dtype=np.float32)
        present = np.zeros((self.n_tok, L), dtype=np.float32)
        for n in range(1, N + 1):
            src_g = g - n
            valid = src_g >= req_start_global
            gather[:, n - 1] = np.where(valid, src_g, 0)
            mask[:, n - 1] = valid.astype(np.float32)
            layer_of = src_g % L             # absolute layer of this slot
            rows = np.nonzero(valid)[0]
            present[rows, layer_of[rows]] = 1.0
        self.gather = torch.from_numpy(gather)
        self.mask = torch.from_numpy(mask)
        self.extra = torch.from_numpy(present)   # (Ntok, L) multi-hot
        self.input_dim = self._input_dim()


# ── Sweep configs ────────────────────────────────────────────────────────────
PAST_TOKEN_WINDOWS = list(range(1, 11))    # W = 1..10
PAST_LAYER_WINDOWS = list(range(1, 25))    # N = 1..24


def make_dataset(mode: str, layer_idx: int, window: int, split: str, data_dir, missing_ok=False):
    if mode == "tokens":
        return PastTokenDataset(layer_idx, window, split, data_dir, missing_ok=missing_ok)
    elif mode == "layers":
        return PastLayerDataset(layer_idx, window, split, data_dir, missing_ok=missing_ok)
    raise ValueError(f"unknown mode '{mode}' (expected 'tokens' or 'layers')")
