"""Evaluate the combined past-layer + offset models on the test split.

For each config (window, offset combo) and each per-layer checkpoint, runs test-set
inference → KL / top-4 overlap / top-4 IoU. Produces:
  summary.csv               — config, layer, kl, overlap, iou, n_samples
  per_layer_overlap.png     — overlap vs layer for each config, with reference lines
  headline table            — each config's mean over layers vs reference baselines.

Usage:
  python evaluate_combined.py --window 5 --offsets-list "1,2,3,4" "1" \
      --seq-store ... --offset-store ... --ckpt-base ... --out-dir ...
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from combined_dataset import CombinedLayerOffsetDataset
from expert_predictor_model import VarDepthMLP, kl_loss, topk_overlap, topk_iou
from train_combined import ckpt_dir_for, NUM_LAYERS_TOTAL

TOPK = 4
# Reference points (test set, averaged over layers) from earlier experiments.
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


def eval_on_test(model, layer, window, combo, seq_store, offset_store, device, batch_size=1024):
    try:
        ds = CombinedLayerOffsetDataset(layer, window, combo, "test", seq_store, offset_store, missing_ok=True)
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


def combo_label(combo):
    return "{" + ",".join(map(str, combo)) + "}"


def load_pastlayer_ref(window: int) -> dict | None:
    """Mean over layers of the past-layer-only result at this window, if available."""
    p = Path("/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/prediction-past_layers/results/summary.csv")
    if not p.exists():
        return None
    kl, ov, iou = [], [], []
    with open(p) as f:
        for r in csv.DictReader(f):
            if int(r["window"]) == window:
                kl.append(float(r["kl"])); ov.append(float(r["overlap"])); iou.append(float(r["iou"]))
    if not ov:
        return None
    return {"kl": float(np.mean(kl)), "overlap": float(np.mean(ov)), "iou": float(np.mean(iou))}


def run(args):
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    combos = [tuple(sorted(int(x) for x in s.split(","))) for s in args.offsets_list]

    rows = []
    print(f"Evaluating combined models — window={args.window} device={device}", flush=True)
    for combo in combos:
        label = combo_label(combo)
        for layer in range(max(combo), NUM_LAYERS_TOTAL):
            ckpt = ckpt_dir_for(args.window, combo, layer, args.ckpt_base) / "best_model.pt"
            model = load_model(ckpt, device)
            if model is None:
                continue
            m = eval_on_test(model, layer, args.window, combo, args.seq_store, args.offset_store, device)
            if m is None:
                continue
            rows.append({"config": f"win{args.window}+{label}", "combo": label, "layer": layer, **m})
            print(f"  {label:<10} layer={layer:2d}  kl={m['kl']:.4f}  "
                  f"overlap={m['overlap']:.4f}  iou={m['iou']:.4f}  n={m['n_samples']}", flush=True)

    if not rows:
        print("No results found. Have you trained any models?")
        return

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("config,combo,layer,kl,overlap,iou,n_samples\n")
        for r in rows:
            f.write(f"{r['config']},{r['combo']},{r['layer']},{r['kl']:.6f},"
                    f"{r['overlap']:.6f},{r['iou']:.6f},{r['n_samples']}\n")
    print(f"\nSaved {csv_path}", flush=True)

    pl_ref = load_pastlayer_ref(args.window)
    _plot(rows, combos, out_dir, args.window, pl_ref)
    _headline(rows, combos, args.window, pl_ref)


def _plot(rows, combos, out_dir, window, pl_ref):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot", flush=True)
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(combos)))
    for combo, c in zip(combos, colors):
        label = combo_label(combo)
        pts = sorted([(r["layer"], r["overlap"]) for r in rows if r["combo"] == label])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=c, label=f"win{window}+{label}", markersize=4)
    ax.axhline(REF_OFFSET1["overlap"], color="gray", ls="--", lw=1, label="offset-1 baseline")
    ax.axhline(REF_ALL4["overlap"], color="green", ls=":", lw=1, label="cross-gate {1,2,3,4}")
    if pl_ref:
        ax.axhline(pl_ref["overlap"], color="purple", ls="-.", lw=1, label=f"past-layer N={window} only")
    ax.set_xlabel("Target layer i"); ax.set_ylabel(f"Test top-{TOPK} overlap")
    ax.set_title(f"Combined past-layer(win{window}) + offset — overlap per layer")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "per_layer_overlap.png", dpi=130); plt.close(fig)
    print(f"Plot written to {out_dir/'per_layer_overlap.png'}", flush=True)


def _headline(rows, combos, window, pl_ref):
    print("\nHEADLINE (test set, averaged over layers):")
    print(f"{'Config':<22}  {'KL':>7}  {'Overlap':>8}  {'IoU':>7}")
    for combo in combos:
        label = combo_label(combo)
        sub = [r for r in rows if r["combo"] == label]
        if not sub:
            continue
        kl = np.mean([r["kl"] for r in sub]); ov = np.mean([r["overlap"] for r in sub])
        iou = np.mean([r["iou"] for r in sub])
        print(f"  win{window}+{label:<14}  {kl:7.4f}  {ov:8.4f}  {iou:7.4f}")
    print("\n  references:")
    print(f"    offset-1        : KL {REF_OFFSET1['kl']:.4f}  overlap {REF_OFFSET1['overlap']:.4f}  IoU {REF_OFFSET1['iou']:.4f}")
    print(f"    cross-gate{{1234}}: KL {REF_ALL4['kl']:.4f}  overlap {REF_ALL4['overlap']:.4f}  IoU {REF_ALL4['iou']:.4f}")
    if pl_ref:
        print(f"    past-layer N={window:<3}: KL {pl_ref['kl']:.4f}  overlap {pl_ref['overlap']:.4f}  IoU {pl_ref['iou']:.4f}")
    else:
        print(f"    past-layer N={window}: (run prediction-past_layers eval to populate this reference)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--offsets-list", nargs="+", default=["1,2,3,4", "1"],
                    help='Comma-separated offset combos, e.g. --offsets-list "1,2,3,4" "1"')
    ap.add_argument("--seq-store", required=True)
    ap.add_argument("--offset-store", required=True)
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
