"""Combined dataset: past-layer history + cross-gating offset features.

Concatenates, per sample, the feature vectors from two independently-built but
token-aligned stores:
  - PastLayerDataset(layer, window)   from prediction-seq_store/dataset
  - ExpertProbDataset(layer, combo)   from predictions-cross_gating/expert_predictor_dataset

Both stores share an identical file→split map and token ordering, so their per-(layer,
split) samples line up one-to-one (verified: targets are element-wise equal). This class
re-asserts that guarantee at construction time.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from seq_dataset import PastLayerDataset
from expert_predictor_dataset import ExpertProbDataset


class CombinedLayerOffsetDataset(Dataset):
    def __init__(
        self,
        layer_idx: int,
        window: int,
        combo: Sequence[int],
        split: str,
        seq_store: str,
        offset_store: str,
        missing_ok: bool = False,
    ) -> None:
        self.layer_idx = layer_idx
        self.window = window
        self.combo = tuple(combo)
        self.split = split

        self.pl = PastLayerDataset(layer_idx, window, split, seq_store, missing_ok=missing_ok)
        self.off = ExpertProbDataset(layer_idx, combo, split, offset_store, missing_ok=missing_ok)

        n_pl, n_off = len(self.pl), len(self.off)
        if n_pl != n_off:
            raise ValueError(
                f"sample-count mismatch for layer {layer_idx} split '{split}': "
                f"past-layer={n_pl} vs offset={n_off} — stores are not aligned."
            )
        # Cheap alignment guard: targets must match (max abs diff 0 when aligned).
        if n_pl > 0:
            diff = (self.pl.y - self.off.y).abs().max().item()
            if diff > 1e-4:
                raise ValueError(
                    f"target mismatch (max abs diff {diff:.3e}) for layer {layer_idx} "
                    f"split '{split}' — stores are not token-aligned."
                )

        self.n_samples = n_pl
        self.input_dim = self.pl.input_dim + self.off.input_dim
        self.num_experts = 60

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int):
        pl_x, y = self.pl[i]
        off_x, _ = self.off[i]
        return torch.cat([pl_x, off_x]), y
