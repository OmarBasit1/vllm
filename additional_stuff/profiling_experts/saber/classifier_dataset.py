"""Build (feature, label) samples for the SABER token->cluster classifier.

Label  : index of the nearest cluster center (under d) to the token's true path.
Feature: sink+window selection of token ids ending at the target token. Token
         ids are shifted +1 so 0 is reserved as the padding id.

Splitting is done by sequence (never by token) to avoid context leakage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


@dataclass
class FeatureConfig:
    n_sink: int = 4
    n_recent: int = 28

    @property
    def seq_len(self) -> int:
        return self.n_sink + self.n_recent + 1  # + target


def load_centers(clusters_json: Path, num_experts: int, k: int, device: str):
    """Return (centers_bits (C, mem_full) float, center_card (C,) float, C)."""
    data = json.loads(Path(clusters_json).read_text())
    clusters = data["clusters"]
    mem_full = data["model"]["num_experts"] * data["model"]["L_moe"]
    C = len(clusters)
    bits = torch.zeros((C, mem_full), dtype=torch.float32)
    for ci, cl in enumerate(clusters):
        for (layer, expert) in cl["center"]:
            bits[ci, layer * num_experts + expert] = 1.0
    return bits.to(device), bits.sum(1).to(device), C, mem_full


def assign_labels(flat_experts: np.ndarray, k: int, num_experts: int,
                  centers_bits: torch.Tensor, center_card: torch.Tensor,
                  device: str, chunk: int = 4096):
    """Nearest-cluster label per token under d(p,c)=|p u c|-max(|p|,|c|).

    flat_experts: (N, n_slots) int. Returns (labels (N,), nonexact (N,) bool).
    """
    N, n_slots = flat_experts.shape
    mem_full = centers_bits.shape[1]
    path_card = float(n_slots)  # each path has exactly n_slots distinct (layer,expert)
    layer_of = np.arange(n_slots) // k  # slot -> layer
    labels = np.empty(N, dtype=np.int64)
    nonexact = np.empty(N, dtype=bool)
    ct = centers_bits.t().contiguous()  # (mem_full, C)
    for s in range(0, N, chunk):
        block = flat_experts[s : s + chunk]
        b = block.shape[0]
        slots = layer_of[None, :] * num_experts + block  # (b, n_slots) global slot ids
        pbits = torch.zeros((b, mem_full), device=device)
        pbits.scatter_(1, torch.as_tensor(slots, dtype=torch.long, device=device), 1.0)
        inter = pbits @ ct  # (b, C)
        union = path_card + center_card[None, :] - inter
        maxc = torch.clamp(center_card[None, :], min=path_card)
        dist = union - maxc  # (b, C)
        dmin, lbl = dist.min(dim=1)
        labels[s : s + b] = lbl.cpu().numpy()
        nonexact[s : s + b] = (dmin > 0).cpu().numpy()
    return labels, nonexact


def build_dataset(parquet_path: Path, cfg: FeatureConfig, k: int, num_experts: int,
                  centers_bits: torch.Tensor, center_card: torch.Tensor, device: str):
    """Return dict with features (N,S) int32, labels (N,), seq_id (N,), nonexact_frac."""
    t = pq.read_table(str(parquet_path)).to_pydict()
    seq_ids = np.asarray(t["seq_id"], dtype=np.int64)
    positions = np.asarray(t["position"], dtype=np.int64)
    token_ids = np.asarray(t["token_id"], dtype=np.int64)
    flat = np.asarray(t["experts"], dtype=np.int16)  # (N, n_slots)

    labels, nonexact = assign_labels(flat, k, num_experts, centers_bits, center_card, device)

    S = cfg.seq_len
    N = len(token_ids)
    feats = np.zeros((N, S), dtype=np.int32)

    # Group token rows by sequence to reconstruct the id stream per sequence.
    order = np.lexsort((positions, seq_ids))
    seq_ids_o, positions_o, token_ids_o = seq_ids[order], positions[order], token_ids[order]
    # boundaries between sequences
    bounds = np.flatnonzero(np.diff(seq_ids_o)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [N]])
    for st, en in zip(starts, ends):
        ids = token_ids_o[st:en]  # in position order
        L = len(ids)
        for t_idx in range(L):
            row = order[st + t_idx]  # original row index
            f = feats[row]
            ns = min(cfg.n_sink, t_idx)
            if ns > 0:
                f[:ns] = ids[:ns] + 1
            nr = min(cfg.n_recent, t_idx)
            if nr > 0:
                f[cfg.n_sink + (cfg.n_recent - nr) : cfg.n_sink + cfg.n_recent] = \
                    ids[t_idx - nr : t_idx] + 1
            f[-1] = ids[t_idx] + 1  # target

    return {
        "features": feats,
        "labels": labels,
        "seq_id": seq_ids,
        "nonexact_frac": float(nonexact.mean()) if N else 0.0,
    }
