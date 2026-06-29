"""Evaluate all trained models and produce ablation summary tables + plots.

For each (layer, offset_combo) pair that has a trained checkpoint, loads the
best model and runs inference on the test split, computing:
  - KL divergence
  - Top-4 overlap (fraction of real top-4 experts recovered by predicted top-4)
  - Mean IoU of predicted top-4 vs real top-4

Outputs (in --out-dir):
  ablation_summary.csv           — long-format: layer, combo, kl, overlap, iou
  heatmap_kl.png                 — layer × config heatmap of test KL
  heatmap_overlap.png            — layer × config heatmap of test top-4 overlap
  marginal_by_combo.png          — per-combo metrics averaged over layers
  marginal_by_layer.png          — per-layer metrics for each combo

Usage:
  python evaluate_ablation.py [--data-dir ...] [--ckpt-base ...] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from expert_predictor_dataset import (
    ABLATION_CONFIGS,
    ExpertProbDataset,
    min_layer_for_combo,
)
from expert_predictor_model import ExpertPredictorMLP, kl_loss, topk_overlap, topk_iou
from train_expert_predictor import (
    CKPT_BASE,
    DATA_DIR,
    NUM_EXPERTS,
    ckpt_dir_for,
)

OUT_DIR_DEFAULT = "/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_results"
TOPK = 4


def load_checkpoint(ckpt_path: Path, device: torch.device) -> ExpertPredictorMLP | None:
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ExpertPredictorMLP(
        input_dim=ckpt["input_dim"],
        h1=ckpt["h1"],
        h2=ckpt["h2"],
        num_experts=ckpt["num_experts"],
        dropout=0.0,  # no dropout at eval time
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def eval_model_on_test(
    model: ExpertPredictorMLP,
    layer: int,
    combo: tuple[int, ...],
    data_dir: str,
    device: torch.device,
    batch_size: int = 1024,
) -> dict | None:
    try:
        ds = ExpertProbDataset(layer, combo, "test", data_dir)
    except (FileNotFoundError, ValueError):
        return None
    if len(ds) == 0:
        return None

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    total_kl = total_ov = total_iou = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_kl += kl_loss(logits, y).item() * len(x)
            probs = F.softmax(logits, dim=-1)
            total_ov += topk_overlap(probs, y, TOPK).item() * len(x)
            total_iou += topk_iou(probs, y, TOPK).item() * len(x)
            n += len(x)

    if n == 0:
        return None
    return {
        "kl": total_kl / n,
        "overlap": total_ov / n,
        "iou": total_iou / n,
        "n_samples": n,
    }


def combo_label(combo: tuple[int, ...]) -> str:
    return "{" + ",".join(map(str, combo)) + "}"


def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.data_dir) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        num_layers = meta["num_layers"]
    else:
        num_layers = 24

    rows = []
    print(f"Evaluating on test split — device={device}", flush=True)

    for combo in ABLATION_CONFIGS:
        label = combo_label(combo)
        min_l = min_layer_for_combo(combo)
        for layer in range(min_l, num_layers):
            ckpt_path = ckpt_dir_for(layer, combo, args.ckpt_base) / "best_model.pt"
            model = load_checkpoint(ckpt_path, device)
            if model is None:
                print(f"  skip {label} layer={layer} — no checkpoint", flush=True)
                continue

            metrics = eval_model_on_test(model, layer, combo, args.data_dir, device)
            if metrics is None:
                print(f"  skip {label} layer={layer} — no test data", flush=True)
                continue

            rows.append({"layer": layer, "combo": label, **metrics})
            print(
                f"  layer={layer:2d}  combo={label:<12s}  "
                f"kl={metrics['kl']:.4f}  overlap={metrics['overlap']:.4f}  "
                f"iou={metrics['iou']:.4f}  n={metrics['n_samples']}",
                flush=True,
            )

    if not rows:
        print("No results found. Have you trained any models?", file=sys.stderr)
        return

    # Save CSV
    csv_path = out_dir / "ablation_summary.csv"
    with open(csv_path, "w") as f:
        f.write("layer,combo,kl,overlap,iou,n_samples\n")
        for r in rows:
            f.write(
                f"{r['layer']},{r['combo']},{r['kl']:.6f},"
                f"{r['overlap']:.6f},{r['iou']:.6f},{r['n_samples']}\n"
            )
    print(f"\nSaved {csv_path}", flush=True)

    _make_plots(rows, out_dir, num_layers)
    print(f"Plots written to {out_dir}", flush=True)


def _make_plots(rows: list[dict], out_dir: Path, num_layers: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots", flush=True)
        return

    combos_ordered = [combo_label(c) for c in ABLATION_CONFIGS]
    layers = list(range(num_layers))

    # Build matrices: rows=layers, cols=combos
    kl_mat = np.full((num_layers, len(combos_ordered)), np.nan)
    ov_mat = np.full((num_layers, len(combos_ordered)), np.nan)
    iou_mat = np.full((num_layers, len(combos_ordered)), np.nan)
    combo_idx = {c: i for i, c in enumerate(combos_ordered)}

    for r in rows:
        li = r["layer"]
        ci = combo_idx.get(r["combo"])
        if ci is not None and 0 <= li < num_layers:
            kl_mat[li, ci] = r["kl"]
            ov_mat[li, ci] = r["overlap"]
            iou_mat[li, ci] = r["iou"]

    # ── Heatmaps ────────────────────────────────────────────────────────────
    for mat, fname, title in [
        (kl_mat, "heatmap_kl.png", "Test KL divergence (lower = better)"),
        (ov_mat, "heatmap_overlap.png", f"Test top-{TOPK} overlap (higher = better)"),
        (iou_mat, "heatmap_iou.png", f"Test top-{TOPK} IoU (higher = better)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, aspect="auto", origin="lower",
                       extent=[-0.5, len(combos_ordered) - 0.5, -0.5, num_layers - 0.5])
        ax.set_xticks(range(len(combos_ordered)))
        ax.set_xticklabels(combos_ordered, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Offset combo")
        ax.set_ylabel("Target layer i")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=130)
        plt.close(fig)

    # ── Marginal by combo (averaged over layers) ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, mat, title, better in [
        (axes[0], kl_mat, "Mean test KL", "lower"),
        (axes[1], ov_mat, f"Mean top-{TOPK} overlap", "higher"),
        (axes[2], iou_mat, f"Mean top-{TOPK} IoU", "higher"),
    ]:
        means = np.nanmean(mat, axis=0)
        stds = np.nanstd(mat, axis=0)
        x = range(len(combos_ordered))
        ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(combos_ordered, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{title} ({better} is better)")
        ax.set_xlabel("Offset combo")
    fig.tight_layout()
    fig.savefig(out_dir / "marginal_by_combo.png", dpi=130)
    plt.close(fig)

    # ── Marginal by layer for each combo ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(combos_ordered)))
    for metric, ax, title in [
        ("overlap", axes[0], f"Top-{TOPK} overlap per layer"),
        ("kl", axes[1], "KL divergence per layer"),
    ]:
        mat = ov_mat if metric == "overlap" else kl_mat
        for ci, (c_label, color) in enumerate(zip(combos_ordered, colors)):
            col = mat[:, ci]
            valid_layers = [l for l in layers if not np.isnan(col[l])]
            valid_vals = [col[l] for l in valid_layers]
            if valid_layers:
                ax.plot(valid_layers, valid_vals, "o-", label=c_label, color=color, markersize=4)
        ax.set_xlabel("Target layer i")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "marginal_by_layer.png", dpi=130)
    plt.close(fig)

    # ── Print headline summary ───────────────────────────────────────────────
    print("\nHEADLINE (test set, averaged over layers):")
    print(f"{'Combo':<14}  {'KL':>7}  {'Overlap':>8}  {'IoU':>7}")
    for ci, c_label in enumerate(combos_ordered):
        kl = np.nanmean(kl_mat[:, ci])
        ov = np.nanmean(ov_mat[:, ci])
        iou = np.nanmean(iou_mat[:, ci])
        print(f"  {c_label:<12}  {kl:7.4f}  {ov:8.4f}  {iou:7.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--ckpt-base", default=CKPT_BASE)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
