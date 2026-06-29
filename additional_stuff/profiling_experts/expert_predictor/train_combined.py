"""Train + small hparam search for the combined past-layer + offset experiment.

One model per (config, layer i). Config = (window, offset combo). Predicts the 60-dim
softmax at (token t, layer i) from concatenated [past-layer window features || offset
features]. Variable-depth MLP (num_layers 2-4).

Usage:
  # hparam search on one cell:
  python train_combined.py --offsets 1 2 3 4 --window 5 --layer 12 \
      --seq-store ... --offset-store ... --ckpt-base ... --hparam-search

  # train all valid layers (called by run_combined.sh):
  python train_combined.py --offsets 1 --window 5 --all-layers \
      --seq-store ... --offset-store ... --ckpt-base ... --shared-params best.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from combined_dataset import CombinedLayerOffsetDataset
from expert_predictor_model import VarDepthMLP, kl_loss
# Reuse the generic evaluation loop and search space from the seq trainer.
from train_seq_predictor import (
    evaluate,
    SEARCH_SPACE,
    FIXED_BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOP_PATIENCE,
    N_TRIALS_DEFAULT,
)

NUM_EXPERTS = 60
NUM_LAYERS_TOTAL = 24
HPARAM_SEARCH_LAYERS = [6, 12, 18]


# ── Loaders ──────────────────────────────────────────────────────────────────

def make_loaders(layer, window, combo, seq_store, offset_store, batch_size, num_workers=2):
    train_ds = CombinedLayerOffsetDataset(layer, window, combo, "train", seq_store, offset_store)
    val_ds = CombinedLayerOffsetDataset(layer, window, combo, "val", seq_store, offset_store, missing_ok=True)
    test_ds = CombinedLayerOffsetDataset(layer, window, combo, "test", seq_store, offset_store, missing_ok=True)
    input_dim = train_ds.input_dim

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, drop_last=False)

    return _dl(train_ds, True), _dl(val_ds, False), _dl(test_ds, False), input_dim


def train_model(
    layer, window, combo, h1, h2, num_layers, lr, batch_size, dropout,
    seq_store, offset_store, device, max_epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE,
    ckpt_path=None, verbose=True,
) -> dict[str, Any]:
    train_dl, val_dl, _, input_dim = make_loaders(
        layer, window, combo, seq_store, offset_store, batch_size)

    model = VarDepthMLP(input_dim, h1, h2, num_layers, NUM_EXPERTS, dropout).to(device)
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
            val_kl = train_kl

        log.append({"epoch": epoch, "train_loss": train_kl, **val_metrics})

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train_loss={train_kl:.4f}"
                  f"  val_kl={val_kl:.4f}  val_overlap={val_metrics['overlap']:.4f}", flush=True)

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
                "h1": h1, "h2": h2, "num_layers": num_layers,
                "dropout": dropout, "num_experts": NUM_EXPERTS,
                "combo": list(combo), "window": window, "layer": layer,
                "best_epoch": best_epoch, "best_val_kl": best_val_kl,
            },
            ckpt_path,
        )

    return {"best_val_kl": best_val_kl, "best_epoch": best_epoch, "log": log}


# ── Hparam search ────────────────────────────────────────────────────────────

def run_hparam_search(layer, window, combo, seq_store, offset_store, ckpt_dir, device, n_trials) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        use_optuna = True
    except ImportError:
        print("  Optuna not found — falling back to grid search.", flush=True)
        use_optuna = False

    def _train(h1, h2, num_layers, lr, dropout):
        return train_model(
            layer, window, combo, h1=h1, h2=h2, num_layers=num_layers, lr=lr,
            batch_size=FIXED_BATCH_SIZE, dropout=dropout,
            seq_store=seq_store, offset_store=offset_store, device=device,
            max_epochs=50, patience=7, ckpt_path=None, verbose=False,
        )["best_val_kl"]

    if use_optuna:
        def objective(trial):
            return _train(
                trial.suggest_categorical("h1", SEARCH_SPACE["h1"]),
                trial.suggest_categorical("h2", SEARCH_SPACE["h2"]),
                trial.suggest_categorical("num_layers", SEARCH_SPACE["num_layers"]),
                trial.suggest_categorical("lr", SEARCH_SPACE["lr"]),
                trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"]),
            )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best["batch_size"] = FIXED_BATCH_SIZE
        best["best_val_kl"] = study.best_value
    else:
        keys = list(SEARCH_SPACE.keys())
        best_kl = float("inf"); best = {}
        for vals in itertools.product(*SEARCH_SPACE.values()):
            p = dict(zip(keys, vals))
            kl = _train(p["h1"], p["h2"], p["num_layers"], p["lr"], p["dropout"])
            if kl < best_kl:
                best_kl = kl; best = dict(p)
        best["batch_size"] = FIXED_BATCH_SIZE
        best["best_val_kl"] = best_kl

    print(f"  hparam search done: best_val_kl={best['best_val_kl']:.4f}  params={best}", flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    return best


def load_or_default_params(ckpt_dir: Path, shared_params: Path | None, args) -> dict:
    for p in (ckpt_dir / "best_params.json", shared_params):
        if p is not None and p.exists():
            print(f"  loaded hparams from {p}", flush=True)
            return json.loads(p.read_text())
    return {
        "h1": args.h1, "h2": args.h2, "num_layers": args.num_layers,
        "lr": args.lr, "batch_size": args.batch_size, "dropout": args.dropout,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def ckpt_dir_for(window: int, combo: tuple[int, ...], layer: int, base: str) -> Path:
    combo_str = "_".join(map(str, combo))
    return Path(base) / f"combined_win{window:02d}_off_{combo_str}_layer_{layer:02d}"


def run_one(args, layer: int, combo: tuple[int, ...]) -> None:
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if layer < max(combo):
        print(f"  skip layer {layer} for combo {combo} (need layer >= {max(combo)})", flush=True)
        return
    ckpt_dir = ckpt_dir_for(args.window, combo, layer, args.ckpt_base)
    print(f"\n{'='*60}\ncombo={combo}  window={args.window}  layer={layer}  device={device}", flush=True)

    if args.hparam_search:
        params = run_hparam_search(
            layer, args.window, combo, args.seq_store, args.offset_store,
            ckpt_dir, device, args.n_trials)
    else:
        shared = Path(args.shared_params) if args.shared_params else None
        params = load_or_default_params(ckpt_dir, shared, args)

    h1 = int(params.get("h1", args.h1))
    h2 = int(params.get("h2", args.h2))
    num_layers = int(params.get("num_layers", args.num_layers))
    lr = float(params.get("lr", args.lr))
    batch_size = int(params.get("batch_size", args.batch_size))
    dropout = float(params.get("dropout", args.dropout))

    ckpt_path = ckpt_dir / "best_model.pt"
    if ckpt_path.exists() and not args.force_retrain:
        print(f"  checkpoint exists: {ckpt_path} — skip (use --force-retrain)", flush=True)
        return

    t0 = time.time()
    result = train_model(
        layer, args.window, combo,
        h1=h1, h2=h2, num_layers=num_layers, lr=lr,
        batch_size=batch_size, dropout=dropout,
        seq_store=args.seq_store, offset_store=args.offset_store, device=device,
        max_epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE,
        ckpt_path=ckpt_path, verbose=True,
    )
    elapsed = time.time() - t0

    (ckpt_dir / "train_log.json").write_text(json.dumps({
        "combo": list(combo), "window": args.window, "layer": layer,
        "h1": h1, "h2": h2, "num_layers": num_layers, "lr": lr,
        "batch_size": batch_size, "dropout": dropout,
        "best_val_kl": result["best_val_kl"], "best_epoch": result["best_epoch"],
        "elapsed_s": elapsed, "epochs": result["log"],
    }, indent=2))
    print(f"  done  best_val_kl={result['best_val_kl']:.4f}  "
          f"epoch={result['best_epoch']}  t={elapsed:.0f}s", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offsets", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--seq-store", required=True)
    ap.add_argument("--offset-store", required=True)
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--shared-params", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hparam-search", action="store_true")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    return ap.parse_args()


def main():
    args = parse_args()
    combo = tuple(sorted(set(args.offsets)))
    if args.all_layers:
        for layer in range(max(combo), NUM_LAYERS_TOTAL):
            run_one(args, layer, combo)
    elif args.layer is not None:
        run_one(args, args.layer, combo)
    else:
        print("ERROR: specify --layer <i> or --all-layers", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
