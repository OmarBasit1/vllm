#!/usr/bin/env python
"""Build the expert-predictor dataset from MoE routing logs.

For each (token, layer_i), computes cross-layer probability vectors by feeding
layer (i-m)'s gate input through layer i's gate weights (for m=1..4), and
saves them alongside the actual expert probabilities as the regression target.

Outputs per-layer NPZ files (split into train/val/test by request/file):
  {out_dir}/layer_{i:02d}_{split}.npz  with keys:
    features  : (N, 4, 60)  float32 — offset probs; NaN where m > i
    targets   : (N, 60)     float32 — actual expert softmax probs at layer i
  {out_dir}/metadata.json
  {out_dir}/split_info.json           — filename -> "train"/"val"/"test"

Usage:
  python dataset_builder.py [--logs-dir ...] [--out-dir ...] [--model ...]
                            [--max-files N] [--token-chunk 256] [--device auto]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Reuse helpers from the sibling script (parent dir is profiling_experts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cross_layer_gate_analysis import (
    _decode,
    load_gate_weights,
    request_decode_batch,
)

MAX_M = 4  # maximum offset to compute

SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def assign_splits(files: list[str]) -> dict[str, str]:
    """Return a filename→split map, splitting by full file path."""
    rng = random.Random(RANDOM_SEED)
    shuffled = sorted(files)  # deterministic order before shuffle
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_FRACS["train"])
    n_val = int(n * SPLIT_FRACS["val"])
    splits = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )
    return {f: s for f, s in zip(shuffled, splits)}


def process_file(
    path: str,
    W: torch.Tensor,
    device: torch.device,
    token_chunk: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (features, targets) for one request file.

    features : (T, MAX_M, E) float32  — cross-layer probs; NaN where m > layer_i
    targets  : (T, L, E)    float32   — actual expert probs per layer
    Note: axis-1 of features is the offset m dimension (0-indexed: m=1..4),
          and the layer dimension is in axis-1 of targets / the per-layer save step.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        record = _decode(zlib.decompress(raw))
    except Exception as e:
        print(f"  WARNING: failed to decode {path}: {e}", flush=True)
        return None

    batch = request_decode_batch(record)
    del record
    if batch is None:
        return None

    G_np, P_np = batch  # G: (T, L, H), P: (T, L, E)
    T, L, H = G_np.shape
    E = P_np.shape[2]
    if T == 0:
        return None

    # features[t, i, m-1, :] = cross-layer prob for target layer i, offset m
    features = np.full((T, L, MAX_M, E), np.nan, dtype=np.float32)

    for m in range(1, MAX_M + 1):
        # target layers i in [m, L-1], source layers j = i-m
        tgt_idx = np.arange(m, L)
        src_idx = tgt_idx - m

        for s in range(0, T, token_chunk):
            G_chunk = torch.from_numpy(G_np[s : s + token_chunk, src_idx, :]).to(device)
            W_tgt = W[tgt_idx]  # (n, E, H)
            # logits[b, n, e] = G_chunk[b, n, :] · W_tgt[n, e, :]
            logits = torch.einsum("bnH,nEH->bnE", G_chunk, W_tgt)
            cross_probs = torch.softmax(logits, dim=-1).cpu().numpy()
            features[s : s + token_chunk, tgt_idx, m - 1, :] = cross_probs

    return features, P_np  # features: (T, L, MAX_M, E), targets: (T, L, E)


def build_dataset(args: argparse.Namespace) -> None:
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"device={device}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, cfg = load_gate_weights(args.model, device)
    L, E, H = W.shape
    print(f"gate weights: L={L} E={E} H={H}", flush=True)

    files = sorted(glob.glob(os.path.join(args.logs_dir, "*.msgpack.zlib")))
    if args.max_files:
        files = files[: args.max_files]
    print(f"files to process: {len(files)}", flush=True)

    split_map = assign_splits(files)
    (out_dir / "split_info.json").write_text(json.dumps(split_map, indent=2))

    # Accumulators: per-layer per-split lists of numpy chunks
    # feat_acc[layer][split] = list of (chunk_T, MAX_M, E) arrays
    feat_acc: list[dict[str, list[np.ndarray]]] = [
        {s: [] for s in ["train", "val", "test"]} for _ in range(L)
    ]
    tgt_acc: list[dict[str, list[np.ndarray]]] = [
        {s: [] for s in ["train", "val", "test"]} for _ in range(L)
    ]
    token_counts: dict[str, int] = defaultdict(int)

    for fi, path in enumerate(files):
        split = split_map[path]
        result = process_file(path, W, device, args.token_chunk)
        if result is None:
            print(f"  [{fi+1}/{len(files)}] skip: {os.path.basename(path)}", flush=True)
            continue
        features, targets = result  # (T, L, MAX_M, E), (T, L, E)
        T = features.shape[0]
        token_counts[split] += T

        for i in range(L):
            feat_acc[i][split].append(features[:, i, :, :])  # (T, MAX_M, E)
            tgt_acc[i][split].append(targets[:, i, :])  # (T, E)

        if (fi + 1) % 50 == 0 or fi == len(files) - 1:
            print(
                f"[{fi+1}/{len(files)}] tokens — "
                + ", ".join(f"{s}:{token_counts[s]}" for s in ["train", "val", "test"]),
                flush=True,
            )

    # valid_offsets[i, m-1] = True if offset m is valid for layer i (i.e. m <= i)
    valid_offsets = np.zeros((L, MAX_M), dtype=bool)
    for i in range(L):
        for m in range(1, MAX_M + 1):
            if m <= i:
                valid_offsets[i, m - 1] = True

    print("Saving per-layer NPZ files...", flush=True)
    for i in range(L):
        for split in ["train", "val", "test"]:
            chunks_f = feat_acc[i][split]
            chunks_t = tgt_acc[i][split]
            if not chunks_f:
                continue
            feat_arr = np.concatenate(chunks_f, axis=0)  # (N, MAX_M, E)
            tgt_arr = np.concatenate(chunks_t, axis=0)   # (N, E)
            out_path = out_dir / f"layer_{i:02d}_{split}.npz"
            np.savez_compressed(
                out_path,
                features=feat_arr,
                targets=tgt_arr,
                valid_offsets=valid_offsets[i],
            )
        if (i + 1) % 6 == 0 or i == L - 1:
            print(f"  saved layers 0..{i}", flush=True)

    metadata = {
        "num_layers": L,
        "num_experts": E,
        "hidden_size": H,
        "max_m": MAX_M,
        "model": args.model,
        "token_counts": dict(token_counts),
        "valid_offsets": valid_offsets.tolist(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Done. Dataset written to {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--logs-dir",
        default="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/all_layers_pre_gating_logs",
    )
    ap.add_argument(
        "--out-dir",
        default="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_dataset",
    )
    ap.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all files")
    ap.add_argument("--token-chunk", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
