"""Train + small hparam search for the past-token / past-layer experiments.

One model per (mode, layer i, window). Predicts the 60-dim softmax at (token t, layer i)
from causal history features (see seq_dataset.py). Variable-depth MLP (num_layers 2-4).

Usage:
  # hparam search on one cell:
  python train_seq_predictor.py --mode tokens --layer 12 --window 5 --store-dir ... \
      --ckpt-base ... --hparam-search

  # train one cell (reads best_params.json if present):
  python train_seq_predictor.py --mode tokens --layer 12 --window 5 --store-dir ... --ckpt-base ...

  # train all layers for a window (called by the run scripts):
  python train_seq_predictor.py --mode layers --all-layers --window 8 --store-dir ... --ckpt-base ...
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from seq_dataset import make_dataset
from expert_predictor_model import VarDepthMLP, kl_loss, topk_overlap, topk_iou

NUM_EXPERTS = 60
NUM_LAYERS_TOTAL = 24
MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 12
N_TRIALS_DEFAULT = 20

# Small hparam search space (num_layers = MLP depth, the new hyperparameter).
SEARCH_SPACE = {
    "h1": [128, 256],
    "h2": [64, 128],
    "num_layers": [2, 3, 4],
    "lr": [5e-4, 1e-3],
    "dropout": [0.0, 0.1],
}
FIXED_BATCH_SIZE = 512
# Representative cells used for hparam search (best config reused across the sweep).
HPARAM_SEARCH_LAYERS = [6, 12, 18]


# ── Loaders ──────────────────────────────────────────────────────────────────

def make_loaders(mode, layer, window, store_dir, batch_size, num_workers=2):
    train_ds = make_dataset(mode, layer, window, "train", store_dir)
    val_ds = make_dataset(mode, layer, window, "val", store_dir, missing_ok=True)
    test_ds = make_dataset(mode, layer, window, "test", store_dir, missing_ok=True)
    input_dim = train_ds.input_dim

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, drop_last=False)

    return _dl(train_ds, True), _dl(val_ds, False), _dl(test_ds, False), input_dim


def evaluate(model, loader, device) -> dict:
    model.eval()
    tot_kl = tot_ov = tot_iou = 0.0
    nb = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot_kl += kl_loss(logits, y).item()
            probs = F.softmax(logits, dim=-1)
            tot_ov += topk_overlap(probs, y).item()
            tot_iou += topk_iou(probs, y).item()
            nb += 1
    if nb == 0:
        return {"kl": float("nan"), "overlap": float("nan"), "iou": float("nan")}
    return {"kl": tot_kl / nb, "overlap": tot_ov / nb, "iou": tot_iou / nb}


def train_model(
    mode, layer, window, h1, h2, num_layers, lr, batch_size, dropout,
    store_dir, device, max_epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE,
    ckpt_path=None, verbose=True,
) -> dict[str, Any]:
    train_dl, val_dl, _, input_dim = make_loaders(mode, layer, window, store_dir, batch_size)

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
                "mode": mode, "layer": layer, "window": window,
                "best_epoch": best_epoch, "best_val_kl": best_val_kl,
            },
            ckpt_path,
        )

    return {"best_val_kl": best_val_kl, "best_epoch": best_epoch, "log": log}


# ── Hparam search ────────────────────────────────────────────────────────────

def _grid_search(mode, layer, window, store_dir, device) -> dict:
    keys = list(SEARCH_SPACE.keys())
    total = 1
    for v in SEARCH_SPACE.values():
        total *= len(v)
    print(f"  grid search: {total} combinations", flush=True)
    best_kl = float("inf")
    best_params: dict = {}
    for i, vals in enumerate(itertools.product(*SEARCH_SPACE.values())):
        params = dict(zip(keys, vals))
        result = train_model(
            mode, layer, window,
            h1=params["h1"], h2=params["h2"], num_layers=params["num_layers"],
            lr=params["lr"], batch_size=FIXED_BATCH_SIZE, dropout=params["dropout"],
            store_dir=store_dir, device=device,
            max_epochs=50, patience=7, ckpt_path=None, verbose=False,
        )
        if result["best_val_kl"] < best_kl:
            best_kl = result["best_val_kl"]
            best_params = dict(params)
            print(f"  [{i+1}/{total}] new best: kl={best_kl:.4f} params={best_params}", flush=True)
    best_params["batch_size"] = FIXED_BATCH_SIZE
    best_params["best_val_kl"] = best_kl
    return best_params


def run_hparam_search(mode, layer, window, store_dir, ckpt_dir, device, n_trials) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        use_optuna = True
    except ImportError:
        print("  Optuna not found — falling back to grid search.", flush=True)
        use_optuna = False

    if use_optuna:
        def objective(trial):
            h1 = trial.suggest_categorical("h1", SEARCH_SPACE["h1"])
            h2 = trial.suggest_categorical("h2", SEARCH_SPACE["h2"])
            num_layers = trial.suggest_categorical("num_layers", SEARCH_SPACE["num_layers"])
            lr = trial.suggest_categorical("lr", SEARCH_SPACE["lr"])
            dropout = trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"])
            result = train_model(
                mode, layer, window,
                h1=h1, h2=h2, num_layers=num_layers, lr=lr,
                batch_size=FIXED_BATCH_SIZE, dropout=dropout,
                store_dir=store_dir, device=device,
                max_epochs=50, patience=7, ckpt_path=None, verbose=False,
            )
            return result["best_val_kl"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best["batch_size"] = FIXED_BATCH_SIZE
        best["best_val_kl"] = study.best_value
    else:
        best = _grid_search(mode, layer, window, store_dir, device)

    print(f"  hparam search done: best_val_kl={best['best_val_kl']:.4f}  params={best}", flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    return best


def load_or_default_params(ckpt_dir: Path, shared_params: Path | None, args) -> dict:
    """Prefer this cell's best_params.json, then a shared params file, then CLI defaults."""
    for p in (ckpt_dir / "best_params.json", shared_params):
        if p is not None and p.exists():
            print(f"  loaded hparams from {p}", flush=True)
            return json.loads(p.read_text())
    return {
        "h1": args.h1, "h2": args.h2, "num_layers": args.num_layers,
        "lr": args.lr, "batch_size": args.batch_size, "dropout": args.dropout,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def ckpt_dir_for(mode: str, layer: int, window: int, base: str) -> Path:
    return Path(base) / f"{mode}_layer_{layer:02d}_win_{window:02d}"


def run_one(args, layer: int, window: int) -> None:
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    ckpt_dir = ckpt_dir_for(args.mode, layer, window, args.ckpt_base)
    print(f"\n{'='*60}\nmode={args.mode}  layer={layer}  window={window}  device={device}", flush=True)

    if args.hparam_search:
        params = run_hparam_search(args.mode, layer, window, args.store_dir, ckpt_dir, device, args.n_trials)
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
        args.mode, layer, window,
        h1=h1, h2=h2, num_layers=num_layers, lr=lr,
        batch_size=batch_size, dropout=dropout,
        store_dir=args.store_dir, device=device,
        max_epochs=MAX_EPOCHS, patience=EARLY_STOP_PATIENCE,
        ckpt_path=ckpt_path, verbose=True,
    )
    elapsed = time.time() - t0

    (ckpt_dir / "train_log.json").write_text(json.dumps({
        "mode": args.mode, "layer": layer, "window": window,
        "h1": h1, "h2": h2, "num_layers": num_layers, "lr": lr,
        "batch_size": batch_size, "dropout": dropout,
        "best_val_kl": result["best_val_kl"], "best_epoch": result["best_epoch"],
        "elapsed_s": elapsed, "epochs": result["log"],
    }, indent=2))
    print(f"  done  best_val_kl={result['best_val_kl']:.4f}  "
          f"epoch={result['best_epoch']}  t={elapsed:.0f}s", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["tokens", "layers"], required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--window", type=int, required=True, help="W (past tokens) or N (past layers)")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--ckpt-base", required=True)
    ap.add_argument("--shared-params", default=None,
                    help="Optional best_params.json reused across the sweep")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hparam-search", action="store_true")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--force-retrain", action="store_true")
    # Fallback hyperparams
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.all_layers:
        for layer in range(NUM_LAYERS_TOTAL):
            run_one(args, layer, args.window)
    elif args.layer is not None:
        run_one(args, args.layer, args.window)
    else:
        print("ERROR: specify --layer <i> or --all-layers", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
