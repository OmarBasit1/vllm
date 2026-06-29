#!/usr/bin/env python
"""Build the shared compact top-k store for the past-token / past-layer experiments.

Both experiments predict the actual 60-dim expert distribution at (token t, layer i),
but use different *causal* history features derived from the logged routing probs.
The underlying data they need is identical, so this builder produces ONE compact store
(top-k indices + values per (token, layer), plus the full-softmax targets) which both
PastTokenDataset and PastLayerDataset read from.

No gate weights are needed here — only the logged `expert_probabilities`.

Per split (train/val/test, split by request/file, seed 42, 70/15/15 — same scheme as
dataset_builder.py), saves:
  seq_{split}_targets.npy   : (Ntok, L, E) float32 — full softmax probs (labels),
                               UNCOMPRESSED so the Dataset can mmap it and slice just
                               layer i (avoids 24 parallel processes each loading ~1.3GB).
  seq_{split}_aux.npz       : topk_idx (Ntok,L,K) int16, topk_val (Ntok,L,K) float32,
                               req_starts (num_requests+1,) int64 — request boundaries.
Plus metadata.json.

Usage:
  python seq_dataset_builder.py [--logs-dir ...] [--out-dir ...] [--max-files N]
                                [--topk 4] [--device auto]
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
from cross_layer_gate_analysis import _decode, request_decode_batch

SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
SPLITS = ("train", "val", "test")


def assign_splits(files: list[str]) -> dict[str, str]:
    """Return a filename→split map, splitting by full file path (same as dataset_builder)."""
    rng = random.Random(RANDOM_SEED)
    shuffled = sorted(files)
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


def process_file(path: str, topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (topk_idx, topk_val, targets) for one request file, or None on failure.

    topk_idx : (T, L, K) int16
    topk_val : (T, L, K) float32
    targets  : (T, L, E) float32
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

    _G, P_np = batch  # P: (T, L, E); gate inputs unused here
    T, L, E = P_np.shape
    if T == 0:
        return None

    P = torch.from_numpy(P_np)
    vals, idx = torch.topk(P, topk, dim=-1)  # (T, L, K)
    topk_idx = idx.to(torch.int16).numpy()
    topk_val = vals.to(torch.float32).numpy()
    return topk_idx, topk_val, P_np


def build_dataset(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.logs_dir, "*.msgpack.zlib")))
    if args.max_files:
        files = files[: args.max_files]
    print(f"files to process: {len(files)}", flush=True)

    split_map = assign_splits(files)
    (out_dir / "split_info.json").write_text(json.dumps(split_map, indent=2))

    # Per-split accumulators
    idx_acc: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    val_acc: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    tgt_acc: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    req_starts: dict[str, list[int]] = {s: [0] for s in SPLITS}  # boundaries in token axis
    token_counts: dict[str, int] = defaultdict(int)

    L = E = None
    for fi, path in enumerate(files):
        split = split_map[path]
        result = process_file(path, args.topk)
        if result is None:
            print(f"  [{fi+1}/{len(files)}] skip: {os.path.basename(path)}", flush=True)
            continue
        topk_idx, topk_val, targets = result
        T, L, E = targets.shape

        idx_acc[split].append(topk_idx)
        val_acc[split].append(topk_val)
        tgt_acc[split].append(targets)
        req_starts[split].append(req_starts[split][-1] + T)
        token_counts[split] += T

        if (fi + 1) % 50 == 0 or fi == len(files) - 1:
            print(
                f"[{fi+1}/{len(files)}] tokens — "
                + ", ".join(f"{s}:{token_counts[s]}" for s in SPLITS),
                flush=True,
            )

    if L is None:
        print("ERROR: no files produced usable data.", file=sys.stderr)
        sys.exit(1)

    print("Saving per-split files...", flush=True)
    for split in SPLITS:
        if not idx_acc[split]:
            print(f"  split '{split}' empty — skipping", flush=True)
            continue
        targets = np.concatenate(tgt_acc[split], axis=0)  # (Ntok, L, E) float32
        # Uncompressed .npy so the Dataset can mmap and slice a single layer cheaply.
        np.save(out_dir / f"seq_{split}_targets.npy", targets)
        np.savez_compressed(
            out_dir / f"seq_{split}_aux.npz",
            topk_idx=np.concatenate(idx_acc[split], axis=0),     # (Ntok, L, K) int16
            topk_val=np.concatenate(val_acc[split], axis=0),     # (Ntok, L, K) float32
            req_starts=np.asarray(req_starts[split], dtype=np.int64),
        )
        # Free per-split memory before next split.
        idx_acc[split] = val_acc[split] = tgt_acc[split] = []
        del targets
        print(f"  saved split '{split}'  ({token_counts[split]} tokens)", flush=True)

    metadata = {
        "num_layers": int(L),
        "num_experts": int(E),
        "topk": int(args.topk),
        "token_counts": dict(token_counts),
        "logs_dir": args.logs_dir,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Done. Store written to {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--logs-dir",
        default="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/all_layers_pre_gating_logs",
    )
    ap.add_argument("--out-dir", required=True, help="Where to write the compact store NPZs")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all files")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--device", default="auto")  # accepted for CLI symmetry; unused
    args = ap.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
