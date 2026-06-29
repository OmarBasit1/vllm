#!/usr/bin/env python
"""Cross-layer gate-input swap analysis for MoE routing logs.

For a target layer ``i`` and offset ``M``, recompute layer ``i``'s expert
probabilities by feeding layer ``i-M``'s logged gate input through layer ``i``'s
gate weights, and compare to layer ``i``'s own recomputed probabilities.

Metrics (per target layer x offset M, averaged over decode tokens / requests):
  * top-4 expert overlap
  * KL divergence  KL(actual_i || crossed_{i,M})

Inputs : per-request ``*.msgpack.zlib`` logs (with per-layer ``gate_input`` and
         ``expert_probabilities``) + the model's per-layer gate weights.
Outputs: npz / csv tables + heatmaps under ``--out-dir``.

Standalone; run in the ``vllm-moe`` conda env. No vLLM serving needed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch

try:
    import msgspec

    def _decode(buf: bytes):
        return msgspec.msgpack.decode(buf)
except Exception:  # pragma: no cover
    import msgpack

    def _decode(buf: bytes):
        return msgpack.unpackb(buf, raw=False)


def find_snapshot(model: str) -> Path:
    safe = "models--" + model.replace("/", "--")
    base = Path(os.path.expanduser("~/.cache/huggingface/hub")) / safe / "snapshots"
    snaps = sorted(base.glob("*"))
    if not snaps:
        raise FileNotFoundError(f"No cached snapshot for {model} under {base}")
    return snaps[-1]


def load_gate_weights(model: str, device: torch.device) -> tuple[torch.Tensor, dict]:
    """Load only the per-layer gate weights via the safetensors index.

    Returns W of shape (num_layers, num_experts, hidden) in fp32 and the config.
    """
    from safetensors import safe_open

    snap = find_snapshot(model)
    cfg = json.loads((snap / "config.json").read_text())
    num_layers = cfg["num_hidden_layers"]
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    weights: list[torch.Tensor | None] = [None] * num_layers
    # Group needed tensors by shard to open each shard once.
    by_shard: dict[str, list[int]] = {}
    for i in range(num_layers):
        name = f"model.layers.{i}.mlp.gate.weight"
        shard = weight_map[name]
        by_shard.setdefault(shard, []).append(i)

    for shard, layers in by_shard.items():
        with safe_open(str(snap / shard), framework="pt") as f:
            for i in layers:
                w = f.get_tensor(f"model.layers.{i}.mlp.gate.weight")
                weights[i] = w.to(torch.float32)

    W = torch.stack(weights, dim=0).to(device)  # (L, E, H)
    return W, cfg


def request_decode_batch(record: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract decode-only (gate_input, logged_probs) for one request.

    Returns (G, P): G (T, L, H) gate inputs, P (T, L, E) logged probabilities.
    Decode = every iteration after the first (prefill) one.
    """
    iters = record.get("moe_expert_activation")
    if not iters or len(iters) < 2:
        return None

    g_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    for it in iters[1:]:  # skip prefill
        layers = sorted(it["layers"], key=lambda L: L["layer_no"])
        if any(L.get("gate_input") is None for L in layers):
            return None
        gi = np.asarray([L["gate_input"] for L in layers], dtype=np.float32)  # (L,tc,H)
        pp = np.asarray(
            [L["expert_probabilities"] for L in layers], dtype=np.float32
        )  # (L,tc,E)
        g_list.append(gi.transpose(1, 0, 2))  # (tc,L,H)
        p_list.append(pp.transpose(1, 0, 2))  # (tc,L,E)

    G = np.concatenate(g_list, axis=0)
    P = np.concatenate(p_list, axis=0)
    return G, P


def topk_mask(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean (..., E) mask of the top-k experts along the last axis."""
    idx = torch.topk(probs, k, dim=-1).indices
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs-dir",
        default="/export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/all_layers_pre_gating_logs",
    )
    ap.add_argument(
        "--out-dir",
        default="/export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/cross_layer_gate_analysis",
    )
    ap.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--max-M", type=int, default=0, help="0 = num_layers-1")
    ap.add_argument("--token-chunk", type=int, default=64)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument(
        "--pred-topk",
        type=int,
        default=0,
        help="experts selected by the crossed PREDICTION (0 = same as --topk; "
        "e.g. k+2 to let the prediction pick more experts). Overlap = "
        "|real_top(k) & pred_top(P)| / k.",
    )
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12

    print(f"device={device}")
    W, cfg = load_gate_weights(args.model, device)
    L, E, H = W.shape
    K = args.topk
    P = args.pred_topk or K  # experts picked by the crossed prediction
    max_M = args.max_M or (L - 1)
    print(
        f"gate weights: layers={L} experts={E} hidden={H}; real_topk={K}; "
        f"pred_topk={P}; max_M={max_M}"
    )

    # Accumulators indexed by (target_layer, M). M index 0 is unused (M>=1).
    shape = (L, L)  # [layer, M]
    sum_kl = np.zeros(shape, dtype=np.float64)
    sumsq_kl = np.zeros(shape, dtype=np.float64)
    sum_ov = np.zeros(shape, dtype=np.float64)
    sumsq_ov = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)

    # Fidelity accumulators (recomputed actual vs logged).
    fid_kl = 0.0
    fid_ov = 0.0
    fid_n = 0

    files = sorted(glob.glob(os.path.join(args.logs_dir, "*.msgpack.zlib")))
    if args.max_files:
        files = files[: args.max_files]
    print(f"processing {len(files)} files")

    diag = torch.arange(L, device=device)

    for fi, path in enumerate(files):
        with open(path, "rb") as fh:
            record = _decode(zlib.decompress(fh.read()))
        batch = request_decode_batch(record)
        del record
        if batch is None:
            continue
        G_np, P_np = batch
        T = G_np.shape[0]

        for s in range(0, T, args.token_chunk):
            G = torch.from_numpy(G_np[s : s + args.token_chunk]).to(device)  # (t,L,H)
            P_logged = torch.from_numpy(P_np[s : s + args.token_chunk]).to(device)
            t = G.shape[0]

            # logits[b, i, j, o] = sum_h G[b,j,h] * W[i,o,h]
            logits = torch.einsum("bjh,ioh->bijo", G, W)
            probs = torch.softmax(logits, dim=-1)  # (t, L, L, E)
            del logits

            # ref[b,i] = probs[b,i,i]  (recomputed actual for layer i)
            ref = probs[:, diag, diag, :]  # (t, L, E)
            ref_mask = topk_mask(ref, K)
            log_ref = torch.log(ref.clamp_min(eps))

            # Fidelity: recomputed actual vs logged.
            log_logged = torch.log(P_logged.clamp_min(eps))
            fkl = (ref * (log_ref - log_logged)).sum(-1)  # (t, L)
            logged_mask = topk_mask(P_logged, K)
            fov = (ref_mask & logged_mask).sum(-1).float() / K
            fid_kl += fkl.sum().item()
            fid_ov += fov.sum().item()
            fid_n += t * L

            # Crossed: for each offset M, target i in [M, L-1], source j=i-M.
            for M in range(1, max_M + 1):
                tgt = torch.arange(M, L, device=device)  # target layers
                src = tgt - M  # source layers
                crossed = probs[:, tgt, src, :]  # (t, n, E)
                r = ref[:, tgt, :]  # (t, n, E)
                lr = log_ref[:, tgt, :]
                lc = torch.log(crossed.clamp_min(eps))
                kl = (r * (lr - lc)).sum(-1)  # (t, n)
                # recovery of the real top-K experts by the predicted top-P set
                ov = (ref_mask[:, tgt, :] & topk_mask(crossed, P)).sum(-1).float() / K

                kl_np = kl.double().sum(0).cpu().numpy()  # (n,)
                klsq_np = (kl.double() ** 2).sum(0).cpu().numpy()
                ov_np = ov.double().sum(0).cpu().numpy()
                ovsq_np = (ov.double() ** 2).sum(0).cpu().numpy()
                tgt_np = tgt.cpu().numpy()
                sum_kl[tgt_np, M] += kl_np
                sumsq_kl[tgt_np, M] += klsq_np
                sum_ov[tgt_np, M] += ov_np
                sumsq_ov[tgt_np, M] += ovsq_np
                count[tgt_np, M] += t

            del probs, ref, ref_mask, log_ref, P_logged, G
        del G_np, P_np
        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"[{fi + 1}/{len(files)}] tokens so far: {int(count.max())}")

    # ----- aggregate -----
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_kl = sum_kl / count
        mean_ov = sum_ov / count
        var_kl = sumsq_kl / count - mean_kl**2
        var_ov = sumsq_ov / count - mean_ov**2
        std_kl = np.sqrt(np.clip(var_kl, 0, None))
        std_ov = np.sqrt(np.clip(var_ov, 0, None))
    # invalid cells (i < M, includes M=0 column) -> NaN
    mean_kl[count == 0] = np.nan
    mean_ov[count == 0] = np.nan
    std_kl[count == 0] = np.nan
    std_ov[count == 0] = np.nan

    fid_kl_mean = fid_kl / max(fid_n, 1)
    fid_ov_mean = fid_ov / max(fid_n, 1)
    print(
        f"\nFIDELITY (recomputed actual vs logged): "
        f"mean KL={fid_kl_mean:.3e}  mean top-{K} overlap={fid_ov_mean:.4f}  "
        f"(n={fid_n})"
    )
    fidelity_ok = fid_ov_mean > 0.99 and fid_kl_mean < 1e-3

    # ----- save -----
    np.savez(
        out_dir / "metrics_by_layer_M.npz",
        mean_kl=mean_kl,
        mean_topk_overlap=mean_ov,
        std_kl=std_kl,
        std_topk_overlap=std_ov,
        count=count,
        topk=K,
        num_layers=L,
        num_experts=E,
    )

    # long-format csv
    with open(out_dir / "metrics_by_layer_M.csv", "w") as f:
        f.write("layer,M,mean_kl,mean_topk_overlap,std_kl,std_topk_overlap,count\n")
        for i in range(L):
            for M in range(1, max_M + 1):
                if count[i, M] == 0:
                    continue
                f.write(
                    f"{i},{M},{mean_kl[i, M]:.6g},{mean_ov[i, M]:.6g},"
                    f"{std_kl[i, M]:.6g},{std_ov[i, M]:.6g},{count[i, M]}\n"
                )

    # marginal over layers (function of M)
    with np.errstate(invalid="ignore"):
        marg_kl = np.nanmean(np.where(count > 0, mean_kl, np.nan), axis=0)
        marg_ov = np.nanmean(np.where(count > 0, mean_ov, np.nan), axis=0)
    with open(out_dir / "marginal_by_M.csv", "w") as f:
        f.write("M,mean_kl,mean_topk_overlap\n")
        for M in range(1, max_M + 1):
            f.write(f"{M},{marg_kl[M]:.6g},{marg_ov[M]:.6g}\n")

    with open(out_dir / "fidelity.txt", "w") as f:
        f.write(
            f"recomputed_actual_vs_logged_mean_kl={fid_kl_mean:.6e}\n"
            f"recomputed_actual_vs_logged_mean_top{K}_overlap={fid_ov_mean:.6f}\n"
            f"n={fid_n}\n"
            f"fidelity_ok={fidelity_ok}\n"
        )

    _make_plots(out_dir, mean_kl, mean_ov, marg_kl, marg_ov, max_M, K, P)

    # headline summary
    print("\nHEADLINE (averaged over layers):")
    print(f"  M=1 : mean top-{K} overlap={marg_ov[1]:.4f}  mean KL={marg_kl[1]:.4f}")
    if max_M >= 2:
        print(f"  M=2 : mean top-{K} overlap={marg_ov[2]:.4f}  mean KL={marg_kl[2]:.4f}")
    print(
        f"  M={max_M}: mean top-{K} overlap={marg_ov[max_M]:.4f}  "
        f"mean KL={marg_kl[max_M]:.4f}"
    )
    print(f"\noutputs written to {out_dir}")
    if not fidelity_ok:
        print(
            "\nWARNING: fidelity check FAILED (recomputed actual does not match logged "
            "probabilities). Inspect gate-weight loading / dtype assumptions before "
            "trusting the crossing results."
        )


def _make_plots(out_dir, mean_kl, mean_ov, marg_kl, marg_ov, max_M, K, P=None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib unavailable, skipping plots: {e}")
        return

    P = P or K
    ov_name = "heatmap_topk_overlap.png" if P == K else f"heatmap_topk_overlap_pred{P}.png"
    ov_title = (
        f"mean top-{K} overlap"
        if P == K
        else f"mean recovery of real top-{K} by predicted top-{P}"
    )
    L = mean_kl.shape[0]
    for arr, name, title in [
        (mean_kl, "heatmap_kl.png", f"mean KL(actual || crossed)"),
        (mean_ov, ov_name, ov_title),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        # columns M=1..max_M, rows layer 0..L-1
        data = arr[:, 1 : max_M + 1]
        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[0.5, max_M + 0.5, -0.5, L - 0.5])
        ax.set_xlabel("offset M (layers back)")
        ax.set_ylabel("target layer i")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    Ms = np.arange(1, max_M + 1)
    ax1.plot(Ms, marg_ov[1 : max_M + 1], "o-", color="tab:blue",
             label=f"top-{K} overlap")
    ax1.set_xlabel("offset M (layers back)")
    ax1.set_ylabel(f"mean top-{K} overlap", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(Ms, marg_kl[1 : max_M + 1], "s-", color="tab:red", label="KL")
    ax2.set_ylabel("mean KL", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("Routing similarity vs offset M (averaged over layers)")
    fig.tight_layout()
    fig.savefig(out_dir / "marginal_by_M.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
