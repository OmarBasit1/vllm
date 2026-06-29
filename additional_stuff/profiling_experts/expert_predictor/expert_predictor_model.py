"""Two-hidden-layer MLP for expert probability prediction.

Architecture:
  Input(input_dim) → Linear(h1) → ReLU → Dropout
                   → Linear(h2) → ReLU → Dropout
                   → Linear(num_experts) → [log-softmax applied at loss time]

Loss: KL divergence between predicted log-probs and target probability vector.
  F.kl_div(F.log_softmax(logits, dim=-1), targets, reduction='batchmean')
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertPredictorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        h1: int,
        h2: int,
        num_experts: int = 60,
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
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (..., num_experts)."""
        return self.net(x)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (no gradients)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


class VarDepthMLP(nn.Module):
    """Variable-depth MLP for sequence-feature expert prediction.

    Hidden widths = [h1] + [h2] * (num_layers - 1), each block
    Linear -> ReLU -> Dropout, followed by a final Linear(last_hidden, num_experts).

    num_layers is the number of HIDDEN layers (2, 3, or 4). For num_layers=2 the
    architecture matches ExpertPredictorMLP(h1, h2).
    """

    def __init__(
        self,
        input_dim: int,
        h1: int,
        h2: int,
        num_layers: int = 2,
        num_experts: int = 60,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers (hidden layers) must be >= 1"
        widths = [h1] + [h2] * (num_layers - 1)
        layers: list[nn.Module] = []
        prev = input_dim
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(dropout)]
            prev = w
        layers.append(nn.Linear(prev, num_experts))
        self.net = nn.Sequential(*layers)
        self.num_experts = num_experts
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (..., num_experts)."""
        return self.net(x)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


def kl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence loss — targets are probability vectors (not log)."""
    return F.kl_div(
        F.log_softmax(logits, dim=-1),
        targets,
        reduction="batchmean",
    )


def topk_overlap(pred_probs: torch.Tensor, target_probs: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Mean fraction of real top-k experts recovered by predicted top-k.

    Returns scalar in [0, 1].
    """
    pred_topk = torch.topk(pred_probs, k, dim=-1).indices  # (N, k)
    real_topk = torch.topk(target_probs, k, dim=-1).indices  # (N, k)

    pred_mask = torch.zeros_like(pred_probs, dtype=torch.bool)
    pred_mask.scatter_(-1, pred_topk, True)
    real_mask = torch.zeros_like(target_probs, dtype=torch.bool)
    real_mask.scatter_(-1, real_topk, True)

    overlap = (pred_mask & real_mask).sum(-1).float() / k  # (N,)
    return overlap.mean()


def topk_iou(pred_probs: torch.Tensor, target_probs: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Mean IoU between predicted top-k set and real top-k set."""
    pred_topk = torch.topk(pred_probs, k, dim=-1).indices
    real_topk = torch.topk(target_probs, k, dim=-1).indices

    pred_mask = torch.zeros_like(pred_probs, dtype=torch.bool)
    pred_mask.scatter_(-1, pred_topk, True)
    real_mask = torch.zeros_like(target_probs, dtype=torch.bool)
    real_mask.scatter_(-1, real_topk, True)

    inter = (pred_mask & real_mask).sum(-1).float()
    union = (pred_mask | real_mask).sum(-1).float()
    iou = inter / union.clamp_min(1.0)
    return iou.mean()
