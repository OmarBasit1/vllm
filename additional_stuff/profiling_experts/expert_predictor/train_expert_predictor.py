"""Train and hyperparameter-search an ExpertPredictorMLP for one (layer, combo).

Workflow
--------
1. Hyperparameter search (Optuna, --n-trials trials) on train+val splits.
   Run only when --hparam-search flag is set.
   Best params are saved to {ckpt_dir}/best_params.json.

2. Final training with best (or provided) hyperparams until convergence,
   with early stopping on val KL divergence.
   Best checkpoint saved to {ckpt_dir}/best_model.pt.
   Training log saved to {ckpt_dir}/train_log.json.

Usage examples
--------------
# Hyperparameter search on layer 8, offset combo (1,2,3,4):
python train_expert_predictor.py --layer 8 --offsets 1 2 3 4 --hparam-search

# Train with found hyperparams (reads best_params.json automatically):
python train_expert_predictor.py --layer 8 --offsets 1 2 3 4

# Train all layers for a given combo (called by run_all.sh):
python train_expert_predictor.py --all-layers --offsets 1 2 3 4

# Provide explicit hyperparams (skip search):
python train_expert_predictor.py --layer 8 --offsets 1 --h1 128 --h2 64 --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from expert_predictor_dataset import (
    ABLATION_CONFIGS,
    ExpertProbDataset,
    min_layer_for_combo,
)
from expert_predictor_model import ExpertPredictorMLP, kl_loss, topk_overlap, topk_iou

# ── Defaults ────────────────────────────────────────────────────────────────
DATA_DIR = "/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_dataset"
CKPT_BASE = "/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_checkpoints"

NUM_EXPERTS = 60
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
N_TRIALS_DEFAULT = 50
HPARAM_SEARCH_LAYERS = [4, 8, 12, 16, 20]  # representative layers for search


# ── Training helpers ─────────────────────────────────────────────────────────

def make_loaders(
    layer: int,
    combo: tuple[int, ...],
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Return (train_dl, val_dl, test_dl, input_dim)."""
    train_ds = ExpertProbDataset(layer, combo, "train", data_dir)
    val_ds = ExpertProbDataset(layer, combo, "val", data_dir, missing_ok=True)
    test_ds = ExpertProbDataset(layer, combo, "test", data_dir, missing_ok=True)
    input_dim = train_ds.input_dim

    def _dl(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return _dl(train_ds, True), _dl(val_ds, False), _dl(test_ds, False), input_dim


def evaluate(model: ExpertPredictorMLP, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_kl = 0.0
    total_ov = 0.0
    total_iou = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_kl += kl_loss(logits, y).item()
            probs = F.softmax(logits, dim=-1)
            total_ov += topk_overlap(probs, y).item()
            total_iou += topk_iou(probs, y).item()
            n_batches += 1
    if n_batches == 0:
        return {"kl": float("nan"), "overlap": float("nan"), "iou": float("nan")}
    return {
        "kl": total_kl / n_batches,
        "overlap": total_ov / n_batches,
        "iou": total_iou / n_batches,
    }


def train_model(
    layer: int,
    combo: tuple[int, ...],
    h1: int,
    h2: int,
    lr: float,
    batch_size: int,
    dropout: float,
    data_dir: str,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    ckpt_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train and return best val metrics + epoch log."""
    train_dl, val_dl, _, input_dim = make_loaders(layer, combo, data_dir, batch_size)

    model = ExpertPredictorMLP(input_dim, h1, h2, NUM_EXPERTS, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_kl = float("inf")
    best_epoch = 0
    best_state = None
    log: list[dict] = []
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = kl_loss(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        train_kl = epoch_loss / max(len(train_dl), 1)
        if len(val_dl.dataset) > 0:
            val_metrics = evaluate(model, val_dl, device)
            val_kl = val_metrics["kl"]
        else:
            val_metrics = {"kl": train_kl, "overlap": float("nan"), "iou": float("nan")}
            val_kl = train_kl  # fall back to train loss when no val set

        log.append({"epoch": epoch, "train_loss": train_kl, **val_metrics})

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(
                f"  epoch {epoch:3d}  train_loss={train_kl:.4f}"
                f"  val_kl={val_kl:.4f}  val_overlap={val_metrics['overlap']:.4f}",
                flush=True,
            )

        if no_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}, best epoch={best_epoch}", flush=True)
            break

    if best_state is not None and ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "input_dim": input_dim,
                "h1": h1,
                "h2": h2,
                "dropout": dropout,
                "num_experts": NUM_EXPERTS,
                "layer": layer,
                "combo": combo,
                "best_epoch": best_epoch,
                "best_val_kl": best_val_kl,
            },
            ckpt_path,
        )

    return {"best_val_kl": best_val_kl, "best_epoch": best_epoch, "log": log}


# ── Hyperparameter search ────────────────────────────────────────────────────

def _grid_search(
    layer: int,
    combo: tuple[int, ...],
    data_dir: str,
    device: torch.device,
) -> dict:
    """Exhaustive grid search fallback when Optuna is unavailable."""
    grid = {
        "h1": [64, 128, 256, 512],
        "h2": [32, 64, 128, 256],
        "lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [256, 512, 1024],
        "dropout": [0.0, 0.1, 0.2],
    }
    import itertools
    best_kl = float("inf")
    best_params: dict = {}
    keys = list(grid.keys())
    total = 1
    for v in grid.values():
        total *= len(v)
    print(f"  grid search: {total} combinations", flush=True)
    for i, vals in enumerate(itertools.product(*grid.values())):
        params = dict(zip(keys, vals))
        result = train_model(
            layer=layer, combo=combo,
            h1=params["h1"], h2=params["h2"], lr=params["lr"],
            batch_size=params["batch_size"], dropout=params["dropout"],
            data_dir=data_dir, device=device,
            max_epochs=60, patience=8, ckpt_path=None, verbose=False,
        )
        if result["best_val_kl"] < best_kl:
            best_kl = result["best_val_kl"]
            best_params = dict(params)
            print(f"  [{i+1}/{total}] new best: kl={best_kl:.4f} params={best_params}", flush=True)
    best_params["best_val_kl"] = best_kl
    return best_params


def run_hparam_search(
    layer: int,
    combo: tuple[int, ...],
    data_dir: str,
    ckpt_dir: Path,
    device: torch.device,
    n_trials: int = N_TRIALS_DEFAULT,
) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        use_optuna = True
    except ImportError:
        print("  Optuna not found — falling back to grid search.", flush=True)
        print("  Install with: pip install optuna  for faster TPE-based search.", flush=True)
        use_optuna = False

    if use_optuna:
        def objective(trial: optuna.Trial) -> float:
            h1 = trial.suggest_categorical("h1", [64, 128, 256, 512])
            h2 = trial.suggest_categorical("h2", [32, 64, 128, 256])
            lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3])
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
            dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
            result = train_model(
                layer=layer, combo=combo,
                h1=h1, h2=h2, lr=lr, batch_size=batch_size, dropout=dropout,
                data_dir=data_dir, device=device,
                max_epochs=60, patience=8, ckpt_path=None, verbose=False,
            )
            return result["best_val_kl"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best["best_val_kl"] = study.best_value
    else:
        best = _grid_search(layer, combo, data_dir, device)

    print(
        f"  hparam search done: best_val_kl={best['best_val_kl']:.4f}  params={best}",
        flush=True,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    return best


def load_or_default_params(ckpt_dir: Path, args: argparse.Namespace) -> dict:
    """Return hyperparams: from best_params.json if it exists, else from CLI args."""
    params_path = ckpt_dir / "best_params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
        print(f"  loaded hparams from {params_path}", flush=True)
        return params
    return {
        "h1": args.h1,
        "h2": args.h2,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def ckpt_dir_for(layer: int, combo: tuple[int, ...], base: str) -> Path:
    combo_str = "_".join(map(str, combo))
    return Path(base) / f"layer_{layer:02d}_combo_{combo_str}"


def run_one(args: argparse.Namespace, layer: int, combo: tuple[int, ...]) -> None:
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    ckpt_dir = ckpt_dir_for(layer, combo, args.ckpt_base)

    # Validate that this combo is feasible for this layer.
    min_l = min_layer_for_combo(combo)
    if layer < min_l:
        print(f"  skip layer {layer} for combo {combo} (need layer >= {min_l})", flush=True)
        return

    print(
        f"\n{'='*60}\nlayer={layer}  combo={combo}  device={device}",
        flush=True,
    )

    if args.hparam_search:
        params = run_hparam_search(
            layer, combo, args.data_dir, ckpt_dir, device, args.n_trials
        )
    else:
        params = load_or_default_params(ckpt_dir, args)

    h1 = params.get("h1", args.h1)
    h2 = params.get("h2", args.h2)
    lr = params.get("lr", args.lr)
    batch_size = params.get("batch_size", args.batch_size)
    dropout = params.get("dropout", args.dropout)

    ckpt_path = ckpt_dir / "best_model.pt"
    if ckpt_path.exists() and not args.force_retrain:
        print(f"  checkpoint already exists: {ckpt_path} — skip (use --force-retrain to override)", flush=True)
        return

    t0 = time.time()
    result = train_model(
        layer=layer,
        combo=combo,
        h1=int(h1), h2=int(h2), lr=float(lr),
        batch_size=int(batch_size), dropout=float(dropout),
        data_dir=args.data_dir,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=EARLY_STOP_PATIENCE,
        ckpt_path=ckpt_path,
        verbose=True,
    )
    elapsed = time.time() - t0

    train_log = {
        "layer": layer,
        "combo": list(combo),
        "h1": h1, "h2": h2, "lr": lr,
        "batch_size": batch_size, "dropout": dropout,
        "best_val_kl": result["best_val_kl"],
        "best_epoch": result["best_epoch"],
        "elapsed_s": elapsed,
        "epochs": result["log"],
    }
    (ckpt_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))
    print(
        f"  done  best_val_kl={result['best_val_kl']:.4f}  "
        f"epoch={result['best_epoch']}  t={elapsed:.0f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=None, help="Target layer (required unless --all-layers)")
    ap.add_argument("--all-layers", action="store_true", help="Train all valid layers for the given combo")
    ap.add_argument("--offsets", type=int, nargs="+", required=True, help="Offset combo, e.g. --offsets 1 2 3 4")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--ckpt-base", default=CKPT_BASE)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hparam-search", action="store_true")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--force-retrain", action="store_true")
    # Fallback hyperparams (used if no best_params.json found)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    combo = tuple(sorted(set(args.offsets)))

    if args.all_layers:
        min_l = min_layer_for_combo(combo)
        # Read num_layers from metadata
        meta_path = Path(args.data_dir) / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            num_layers = meta["num_layers"]
        else:
            num_layers = 24  # fallback
        for layer in range(min_l, num_layers):
            run_one(args, layer, combo)
    elif args.layer is not None:
        run_one(args, args.layer, combo)
    else:
        print("ERROR: specify --layer <i> or --all-layers", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
