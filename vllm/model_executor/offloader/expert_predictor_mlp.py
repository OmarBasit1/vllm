# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLP architecture for the expert-cache "hybrid" predictor.

Mirrors the research/training code's `ExpertPredictorMLP`
(`additional_stuff/profiling_experts/expert_predictor/expert_predictor_model.py`
on the `gate-input-logging` branch) exactly, so checkpoints trained there load
here unmodified. Inference-only: no loss/metric helpers.
"""

import torch
import torch.nn as nn


class ExpertPredictorMLP(nn.Module):
    """Two-hidden-layer MLP predicting a layer's expert routing distribution
    from the previous layer's hidden state."""

    def __init__(
        self,
        input_dim: int,
        h1: int,
        h2: int,
        num_experts: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (..., num_experts)."""
        return self.net(x)
