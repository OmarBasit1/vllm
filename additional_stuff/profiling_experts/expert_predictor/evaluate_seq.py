"""Evaluate trained past-token / past-layer models on the test split and plot results.

For each (layer, window) checkpoint of the given mode, runs test-set inference and reports
KL / top-4 overlap / top-4 IoU. Produces:
  summary.csv               — layer, window, kl, overlap, iou, n_samples
  heatmap_{kl,overlap,iou}.png  — layer × window
  marginal_by_window.png    — metrics vs window size, averaged over layers,
                              with the offset-1 baseline and {1,2,3,4} reference lines.

Usage:
  python evaluate_seq.py --mode tokens --store-dir ... --ckpt-base ... --out-dir ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seq_dataset import make_dataset, PAST_TOKEN_WINDOWS, PAST_LAYER_WINDOWS
from expert_predictor_model import VarDepthMLP, kl_loss, topk_overlap, topk_iou
from train_seq_predictor import ckpt_dir_for, NUM_LAYERS_TOTAL

TOPK = 4
# Reference points from the cross-gating experiment (test set, averaged over layers).
REF_OFFSET1 = {"kl": 0.0431, "overlap": 0.7386, "iou": 0.6342}
REF_ALL4 = {"kl": 0.0335, "overlap": 0.7883, "iou": 0.6900}


def load_model(ckpt_path: Path, device) -> VarDepthMLP | None:
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = VarDepthMLP(
        input_dim=ckpt["input_dim"], h1=ckpt["h1"], h2=ckpt["h2"],
        num_layers=ckpt.get("num_layers", 2), num_experts=ckpt["num_experts"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def eval_on_test(model, mode, layer, window, store_dir, device, batch_size=1024) -> dict | None:
    try:
        ds = make_dataset(mode, layer, window, "test", store_dir, missing_ok=True)
    except (FileNotFoundError, ValueError):
        return None
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    tot_kl = tot_ov = tot_iou = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot_kl += kl_loss(logits, y).item() * len(x)
            probs = F.softmax(logits, dim=-1)
            tot_ov += topk_overlap(probs, y, TOPK).item() * len(x)
            tot_iou += topk_iou(probs, y, TOPK).item() * len(x)
            n += len(x)
    if n == 0:
        return None
    return {"kl": tot_kl / n, "overlap": tot_ov / n, "iou": tot_iou / n, "n_samples": n}


def run(args):
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = PAST_TOKEN_WINDOWS if args.mode == "tokens" else PAST_LAYER_WINDOWS

    rows = []
    print(f"Evaluating mode={args.mode} on test split — device={device}", flush=True)
    for window in windows:
        for layer in range(NUM_LAYERS_TOTAL):
            ckpt = ckpt_dir_for(args.mode, layer, window, args.ckpt_base) / "best_model.pt"
            model = load_model(ckpt, device)
            if model is None:
                continue
            m = eval_on_test(model, args.mode, layer, window, args.store_dir, device)
            if m is None:
                continue
            rows.append({"layer": layer, "window": window, **m})
            print(f"  layer={layer:2d}  win={window:2d}  kl={m['kl']:.4f}  "
                  f"overlap={m['overlap']:.4f}  iou={m['iou']:.4f}  n={m['n_samples']}", flush=True)

    if not rows:
        print("No results found. Have you trained any models?", file=sys.stderr)
        return

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("layer,window,kl,overlap,iou,n_samples\n")
        for r in rows:
            f.write(f"{r['layer']},{r['window']},{r['kl']:.6f},"
                    f"{r['overlap']:.6f},{r['iou']:.6f},{r['n_samples']}\n")
    print(f"\nSaved {csv_path}", flush=True)

    _plots(rows, out_dir, windows, args.mode)
    _headline(rows, windows)


def _matrices(rows, windows):
    win_idx = {w: j for j, w in enumerate(windows)}
    shape = (NUM_LAYERS_TOTAL, len(windows))
    kl = np.full(shape, np.nan); ov = np.full(shape, np.nan); iou = np.full(shape, np.nan)
    for r in rows:
        j = win_idx.get(r["window"])
        li = r["layer"]
        if j is not None and 0 <= li < NUM_LAYERS_TOTAL:
            kl[li, j] = r["kl"]; ov[li, j] = r["overlap"]; iou[li, j] = r["iou"]
    return kl, ov, iou


def _plots(rows, out_dir, windows, mode):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots", flush=True)
        return

    kl, ov, iou = _matrices(rows, windows)
    wlabel = "Past tokens W" if mode == "tokens" else "Past layers N"

    for mat, fname, title in [
        (kl, "heatmap_kl.png", "Test KL (lower=better)"),
        (ov, "heatmap_overlap.png", f"Test top-{TOPK} overlap (higher=better)"),
        (iou, "heatmap_iou.png", f"Test top-{TOPK} IoU (higher=better)"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, len(windows) * 0.5), 6))
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", origin="lower")
        ax.set_xticks(range(len(windows))); ax.set_xticklabels(windows, fontsize=8)
        ax.set_xlabel(wlabel); ax.set_ylabel("Target layer i"); ax.set_title(title)
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130); plt.close(fig)

    # Marginal by window (averaged over layers) with reference lines.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, mat, key, title, better in [
        (axes[0], kl, "kl", "Mean test KL", "lower"),
        (axes[1], ov, "overlap", f"Mean top-{TOPK} overlap", "higher"),
        (axes[2], iou, "iou", f"Mean top-{TOPK} IoU", "higher"),
    ]:
        means = np.nanmean(mat, axis=0)
        ax.plot(windows, means, "o-", color="steelblue", label=mode)
        ax.axhline(REF_OFFSET1[key], color="gray", ls="--", lw=1, label="offset-1 baseline")
        ax.axhline(REF_ALL4[key], color="green", ls=":", lw=1, label="cross-gate {1,2,3,4}")
        ax.set_xlabel(wlabel); ax.set_title(f"{title} ({better} better)")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "marginal_by_window.png", dpi=130); plt.close(fig)
    print(f"Plots written to {out_dir}", flush=True)


def _headline(rows, windows):
    kl, ov, iou = _matrices(rows, windows)
    print("\nHEADLINE (test set, averaged over layers):")
    print(f"{'Window':>7}  {'KL':>7}  {'Overlap':>8}  {'IoU':>7}")
    for j, w in enumerate(windows):
        print(f"  {w:>5}  {np.nanmean(kl[:, j]):7.4f}  "
              f"{np.nanmean(ov[:, j]):8.4f}  {np.nanmean(iou[:, j]):7.4f}")
    print(f"\n  ref offset-1   : KL {REF_OFFSET1['kl']:.4f}  "
          f"overlap {REF_OFFSET1['overlap']:.4f}  IoU {REF_OFFSET1['iou']:.4f}")
    print(f"  ref {{1,2,3,4}}  : KL {REF_ALL4['kl']:.4f}  "
          f"overlap {REF_ALL4['overlap']:.4f}  IoU {REF_ALL4['iou']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["tokens", "layers"], required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
