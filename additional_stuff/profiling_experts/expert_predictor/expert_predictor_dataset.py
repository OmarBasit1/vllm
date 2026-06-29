"""PyTorch Dataset for the expert-predictor task.

Each sample is (x, y) where:
  x : (len(offset_combo) * num_experts,)  float32  — concatenated cross-layer probs
  y : (num_experts,)                       float32  — actual softmax probs at layer i

Samples with NaN features (invalid offset for the given layer) are dropped
automatically at load time so callers never see NaN.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# Canonical ablation configurations (offset subsets) used across scripts.
ABLATION_CONFIGS: list[tuple[int, ...]] = [
    (1,),
    (2,),
    (3,),
    (4,),
    (1, 2),
    (1, 2, 3),
    (1, 2, 3, 4),
]

SPLITS = ("train", "val", "test")


class ExpertProbDataset(Dataset):
    """Load features/targets for a single (layer, offset_combo, split) cell.

    Parameters
    ----------
    layer_idx     : target layer i (0-based)
    offset_combo  : tuple of ints, e.g. (1,) or (1, 2, 3)
    split         : "train", "val", or "test"
    data_dir      : directory produced by dataset_builder.py
    """

    def __init__(
        self,
        layer_idx: int,
        offset_combo: Sequence[int],
        split: str,
        data_dir: str | Path,
        missing_ok: bool = False,
    ) -> None:
        data_dir = Path(data_dir)
        assert split in SPLITS, f"split must be one of {SPLITS}"

        npz_path = data_dir / f"layer_{layer_idx:02d}_{split}.npz"
        if not npz_path.exists():
            if missing_ok:
                self.x = torch.empty(0, 0)
                self.y = torch.empty(0, 0)
                self.input_dim = len(offset_combo) * 60
                self.num_experts = 60
                self.layer_idx = layer_idx
                self.offset_combo = tuple(offset_combo)
                self.split = split
                self.n_samples = 0
                return
            raise FileNotFoundError(npz_path)

        data = np.load(npz_path)
        features_raw = data["features"]   # (N, MAX_M, E)
        targets_raw = data["targets"]     # (N, E)
        valid_offsets = data["valid_offsets"]  # (MAX_M,) bool

        # Validate that all requested offsets are valid for this layer.
        for m in offset_combo:
            if not valid_offsets[m - 1]:
                raise ValueError(
                    f"Offset m={m} is not valid for layer {layer_idx} "
                    f"(need layer >= m). valid_offsets={valid_offsets.tolist()}"
                )

        # Select columns for the requested offsets (0-indexed: m-1).
        m_indices = [m - 1 for m in offset_combo]
        selected = features_raw[:, m_indices, :]  # (N, len(combo), E)

        # Drop samples with any NaN (should not happen after validation, but guard).
        valid_mask = ~np.isnan(selected).any(axis=(1, 2))
        selected = selected[valid_mask]
        targets = targets_raw[valid_mask]

        # Flatten offset × expert dims → (N, len(combo) * E)
        N, n_offsets, E = selected.shape
        self.x = torch.from_numpy(selected.reshape(N, n_offsets * E))
        self.y = torch.from_numpy(targets)
        self.input_dim = n_offsets * E
        self.num_experts = E
        self.layer_idx = layer_idx
        self.offset_combo = tuple(offset_combo)
        self.split = split
        self.n_samples = N

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def load_metadata(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "metadata.json"
    return json.loads(path.read_text())


def min_layer_for_combo(combo: Sequence[int]) -> int:
    """Minimum target layer index for which all offsets in `combo` are valid."""
    return max(combo)
