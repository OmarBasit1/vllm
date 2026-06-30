#!/usr/bin/env python
"""Phase 4 - train the SABER token->cluster classifier.

Trains on a pool of datasets (default: all), splitting BY SEQUENCE into
train/held-out, reports held-out cluster-classification accuracy and per-token
inference latency, and saves classifier.pt. The reusable helpers here are also
imported by evaluate.py for the distribution-shift protocol.

Usage:
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib python train_classifier.py [--epochs 15] \
      [--train-datasets ...] [--val-frac 0.15]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from classifier_dataset import FeatureConfig, build_dataset, load_centers  # noqa: E402
from classifier_model import SinkWindowClassifier  # noqa: E402


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_vocab(model: str) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    return len(tok) + 1  # +1 for the reserved padding id (token ids shifted +1)


def prepare(datasets, cfg, k, num_experts, centers_bits, center_card, device):
    """Concatenate per-dataset samples; track dataset + seq ids for splitting."""
    feats, labels, seq_keys, ds_keys = [], [], [], []
    nonexact = {}
    for di, ds in enumerate(datasets):
        p = common.PATHS_DIR / f"{ds}.parquet"
        if not p.exists():
            print(f"  skip missing {p}", flush=True)
            continue
        d = build_dataset(p, cfg, k, num_experts, centers_bits, center_card, device)
        feats.append(d["features"])
        labels.append(d["labels"])
        # unique sequence key per (dataset, seq_id)
        seq_keys.append(d["seq_id"] + di * 10_000_000)
        ds_keys.append(np.full(len(d["labels"]), di, dtype=np.int64))
        nonexact[ds] = d["nonexact_frac"]
    if not feats:
        raise SystemExit("no datasets found; run capture_paths.py first")
    return (
        np.concatenate(feats), np.concatenate(labels),
        np.concatenate(seq_keys), np.concatenate(ds_keys), nonexact,
    )


def seq_split(seq_keys, val_frac, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq_keys)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * val_frac)
    val_set = set(uniq[:n_val].tolist())
    is_val = np.array([s in val_set for s in seq_keys])
    return ~is_val, is_val


def train_model(feats, labels, num_clusters, vocab, cfg, epochs, device,
                batch=512, lr=1e-3, use_class_weights=True):
    model = SinkWindowClassifier(vocab, num_clusters, seq_len=cfg.seq_len).to(device)
    X = torch.as_tensor(feats, dtype=torch.long)
    y = torch.as_tensor(labels, dtype=torch.long)
    if use_class_weights:
        counts = np.bincount(labels, minlength=num_clusters).astype(np.float64)
        w = 1.0 / np.clip(counts, 1, None)
        w = w / w.sum() * num_clusters
        weights = torch.as_tensor(w, dtype=torch.float32, device=device)
    else:
        weights = None
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            xb = X[idx].to(device)
            yb = y[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"    epoch {ep+1}/{epochs} loss={tot/n:.4f}", flush=True)
    return model


@torch.no_grad()
def accuracy(model, feats, labels, device, batch=2048):
    model.eval()
    X = torch.as_tensor(feats, dtype=torch.long)
    y = torch.as_tensor(labels, dtype=torch.long)
    correct = 0
    for s in range(0, len(y), batch):
        xb = X[s : s + batch].to(device)
        pred = model(xb).argmax(1).cpu()
        correct += (pred == y[s : s + batch]).sum().item()
    return correct / max(len(y), 1)


@torch.no_grad()
def per_token_latency_ms(model, cfg, device, n=2000):
    model.eval()
    x = torch.randint(1, 1000, (1, cfg.seq_len), device=device)
    for _ in range(20):  # warmup
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-datasets", nargs="*", default=common.DATASETS)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--n-recent", type=int, default=28)
    args = ap.parse_args()

    device = device_str()
    mc = json.loads(common.MODEL_CONFIG.read_text())
    k, num_experts = mc["num_experts_per_tok"], mc["num_experts"]
    cfg = FeatureConfig(n_sink=args.n_sink, n_recent=args.n_recent)

    centers_bits, center_card, C, _ = load_centers(common.CLUSTERS, num_experts, k, device)
    vocab = get_vocab(common.MODEL)
    print(f"clusters |C*|={C}; vocab={vocab}; feature seq_len={cfg.seq_len}", flush=True)

    feats, labels, seq_keys, _, nonexact = prepare(
        args.train_datasets, cfg, k, num_experts, centers_bits, center_card, device)
    tr, va = seq_split(seq_keys, args.val_frac)
    print(f"samples: train={tr.sum()} val={va.sum()}; nonexact_frac={nonexact}", flush=True)

    model = train_model(feats[tr], labels[tr], C, vocab, cfg, args.epochs, device)
    acc = accuracy(model, feats[va], labels[va], device)
    lat = per_token_latency_ms(model, cfg, device)
    print(f"\nheld-out accuracy={acc:.4f}  per-token latency={lat:.4f} ms", flush=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_clusters": C,
            "vocab": vocab,
            "cfg": vars(cfg) | {"seq_len": cfg.seq_len},
            "n_sink": cfg.n_sink, "n_recent": cfg.n_recent,
            "held_out_accuracy": acc,
            "latency_ms": lat,
            "nonexact_frac": nonexact,
        },
        common.CLASSIFIER,
    )
    print(f"saved {common.CLASSIFIER}", flush=True)


if __name__ == "__main__":
    main()
