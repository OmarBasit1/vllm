#!/usr/bin/env python
"""Phase 5 - report cluster-prediction accuracy (the deliverable).

Writes eval_report.md with:
  1. Held-out cluster-classification accuracy (train on all, sequence-split test).
  2. Distribution shift (paper Table 3): train on XSUM, test on the others.
  3. Per-token inference latency.
  4. (optional, best-effort) activation-ratio check vs sequential / oracle batching.

Every result is annotated with (L_moe, num_experts, k), tau, |V|, |C*|.

Usage:
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib python evaluate.py [--epochs 15]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from classifier_dataset import FeatureConfig, build_dataset, load_centers  # noqa: E402
from train_classifier import (  # noqa: E402
    accuracy, device_str, get_vocab, per_token_latency_ms, prepare, seq_split, train_model,
)


def activation_ratio(model, cfg, k, num_experts, mem_full, device, dataset, G=32):
    """Best-effort: mean per-batch expert-union fraction under sequential vs
    predicted-cluster vs oracle-cluster batching, on one dataset."""
    p = common.PATHS_DIR / f"{dataset}.parquet"
    if not p.exists():
        return None
    t = pq.read_table(str(p)).to_pydict()
    flat = np.asarray(t["experts"], dtype=np.int16)
    centers_bits, center_card, _, _ = load_centers(common.CLUSTERS, num_experts, k, device)
    d = build_dataset(p, cfg, k, num_experts, centers_bits, center_card, device)
    feats = torch.as_tensor(d["features"], dtype=torch.long)
    true_lbl = d["labels"]
    model.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, len(feats), 4096):
            preds.append(model(feats[s : s + 4096].to(device)).argmax(1).cpu().numpy())
    pred_lbl = np.concatenate(preds)

    layer_of = np.arange(flat.shape[1]) // k

    def ar_for_order(order):
        ars = []
        for s in range(0, len(order), G):
            idx = order[s : s + G]
            slots = layer_of[None, :] * num_experts + flat[idx]
            ars.append(len(np.unique(slots)) / mem_full)
        return float(np.mean(ars)) if ars else float("nan")

    seq_order = np.arange(len(flat))
    pred_order = np.argsort(pred_lbl, kind="stable")
    oracle_order = np.argsort(true_lbl, kind="stable")
    return {
        "dataset": dataset,
        "batch_size": G,
        "sequential": ar_for_order(seq_order),
        "predicted_cluster": ar_for_order(pred_order),
        "oracle_cluster": ar_for_order(oracle_order),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--shift-train", default="xsum")
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--n-recent", type=int, default=28)
    ap.add_argument("--no-ar", action="store_true", help="skip the activation-ratio check")
    args = ap.parse_args()

    device = device_str()
    mc = json.loads(common.MODEL_CONFIG.read_text())
    k, num_experts, L_moe = mc["num_experts_per_tok"], mc["num_experts"], mc["L_moe"]
    mem_full = num_experts * L_moe
    cfg = FeatureConfig(n_sink=args.n_sink, n_recent=args.n_recent)

    vp_rep = json.loads((common.OUT_ROOT / "valid_paths_report.json").read_text())
    cl_rep = json.loads((common.OUT_ROOT / "clusters_report.json").read_text())
    tau, V, Cstar = vp_rep["tau"], vp_rep["valid_paths_after"], cl_rep["num_clusters"]

    centers_bits, center_card, C, _ = load_centers(common.CLUSTERS, num_experts, k, device)
    vocab = get_vocab(common.MODEL)

    avail = [d for d in common.DATASETS if (common.PATHS_DIR / f"{d}.parquet").exists()]
    print(f"available datasets: {avail}", flush=True)

    # 1. Held-out (train on all available, sequence split).
    feats, labels, seq_keys, ds_keys, nonexact = prepare(
        avail, cfg, k, num_experts, centers_bits, center_card, device)
    tr, va = seq_split(seq_keys, 0.15)
    print("training held-out model...", flush=True)
    model = train_model(feats[tr], labels[tr], C, vocab, cfg, args.epochs, device)
    held_out_acc = accuracy(model, feats[va], labels[va], device)
    latency = per_token_latency_ms(model, cfg, device)
    majority = float(np.bincount(labels[va], minlength=C).max() / max(va.sum(), 1))

    # 2. Distribution shift: train on shift-train, test on each other dataset.
    shift = {}
    shift_model = None
    if args.shift_train in avail:
        sd = build_dataset(common.PATHS_DIR / f"{args.shift_train}.parquet",
                           cfg, k, num_experts, centers_bits, center_card, device)
        print(f"training shift model on {args.shift_train}...", flush=True)
        shift_model = train_model(sd["features"], sd["labels"], C, vocab, cfg,
                                  args.epochs, device)
        for ds in avail:
            if ds == args.shift_train:
                continue
            td = build_dataset(common.PATHS_DIR / f"{ds}.parquet",
                               cfg, k, num_experts, centers_bits, center_card, device)
            shift[ds] = accuracy(shift_model, td["features"], td["labels"], device)

    # 3. Optional activation-ratio check.
    ar = None
    if not args.no_ar:
        try:
            ar = activation_ratio(model, cfg, k, num_experts, mem_full, device, avail[0])
        except Exception as e:  # pragma: no cover
            ar = {"error": str(e)}

    # ---- write report ----
    ann = f"(L_moe={L_moe}, num_experts={num_experts}, k={k}), tau={tau}, |V|={V}, |C*|={Cstar}"
    lines = [
        "# SABER expert-prediction recreation - evaluation report",
        "",
        f"Model: `{common.MODEL}`  ·  routing captured from vLLM's real gate + top-k.",
        f"Annotations: {ann}",
        f"Feature: sink+window (n_sink={cfg.n_sink}, n_recent={cfg.n_recent}, "
        f"seq_len={cfg.seq_len}).",
        "",
        "## 1. Held-out cluster-classification accuracy",
        "",
        f"- **accuracy = {held_out_acc:.4f}** (majority-class baseline = {majority:.4f})",
        f"- per-token inference latency = **{latency:.4f} ms**",
        f"- non-exact center fraction (path not a subset of its assigned cluster): "
        f"{json.dumps({d: round(v,4) for d,v in nonexact.items()})}",
        "",
        "## 2. Distribution shift (train on %s -> test on others)" % args.shift_train,
        "",
        "| test dataset | accuracy |",
        "|---|---|",
    ]
    for ds, a in shift.items():
        lines.append(f"| {ds} | {a:.4f} |")
    if not shift:
        lines.append("| (shift-train dataset unavailable) | - |")
    lines += ["", "## 3. Per-token latency", "", f"- {latency:.4f} ms/token "
              f"(target < 1 ms).", ""]
    if ar is not None:
        lines += ["## 4. Activation-ratio check (optional)", "",
                  "Mean fraction of experts activated per batch (lower = better grouping):",
                  "", "```", json.dumps(ar, indent=2), "```", ""]
    common.EVAL_REPORT.write_text("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nwrote {common.EVAL_REPORT}", flush=True)


if __name__ == "__main__":
    main()
