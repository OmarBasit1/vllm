#!/usr/bin/env python
"""Do offsets N and N+1 *together* recover a layer's real top-k routing?

For a target layer ``i``, take the crossed top-k expert predictions obtained by
feeding layer ``i-N``'s and layer ``i-(N+1)``'s gate inputs through layer ``i``'s
gate, union the two predicted sets, and compare against layer ``i``'s real top-k:

  * superset rate : fraction of (token, layer) cases where
                    union(top_k@N, top_k@N+1)  is a SUPERSET of the real top-k
                    (i.e. both offsets combined contain every real expert).
  * IoU           : |union(top_k@N, top_k@N+1)  ∩  real|
                    -------------------------------------------
                    |union(top_k@N, top_k@N+1)  ∪  real|

Produces one figure with two graphs (superset rate vs N, IoU vs N), plus the
per-(layer, N) tables.

Reuses helpers from ``cross_layer_gate_analysis.py`` (same directory).
Standalone offline analysis; run in the ``vllm-moe`` conda env.
"""

from __future__ import annotations

import argparse
import glob
import os
import zlib
from pathlib import Path

import numpy as np
import torch

from cross_layer_gate_analysis import (
    _decode,
    load_gate_weights,
    request_decode_batch,
    topk_mask,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs-dir",
        default="/export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/all_layers_pre_gating_logs",
    )
    ap.add_argument(
        "--out-dir",
        default="/export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/topk_union_offset_analysis",
    )
    ap.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--token-chunk", type=int, default=64)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument(
        "--real",
        choices=["logged", "recomputed"],
        default="logged",
        help="reference 'real' top-k: logged probabilities vs recomputed self (offset 0)",
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

    print(f"device={device}")
    W, cfg = load_gate_weights(args.model, device)
    L, E, H = W.shape
    K = args.topk
    # offsets N where both N and N+1 are valid for some target layer:
    #   need i >= N+1  ->  N in [1, L-2]
    max_N = L - 2
    print(f"layers={L} experts={E} hidden={H}; topk={K}; N in [1,{max_N}]; real={args.real}")

    diag = torch.arange(L, device=device)

    # Accumulators indexed by (target_layer, N). N index 0 unused.
    shape = (L, L)
    sum_superset = np.zeros(shape, dtype=np.float64)
    sum_iou = np.zeros(shape, dtype=np.float64)
    sumsq_iou = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)

    # Single-offset baseline indexed by (target_layer, M), M in [1, L-1]:
    # does a SINGLE offset's top-k prediction cover the real top-k
    # (for equal-size sets this is an exact top-k match).
    max_M_single = L - 1
    sum_superset_single = np.zeros(shape, dtype=np.float64)
    count_single = np.zeros(shape, dtype=np.int64)

    files = sorted(glob.glob(os.path.join(args.logs_dir, "*.msgpack.zlib")))
    if args.max_files:
        files = files[: args.max_files]
    print(f"processing {len(files)} files")

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
            t = G.shape[0]

            logits = torch.einsum("bjh,ioh->bijo", G, W)
            probs = torch.softmax(logits, dim=-1)  # (t, i, j, E)
            del logits
            tk = topk_mask(probs, K)  # (t, i, j, E) bool — top-k per (target,source)

            if args.real == "recomputed":
                real_mask = tk[:, diag, diag, :]  # diagonal (t, i, E)
            else:
                P_logged = torch.from_numpy(P_np[s : s + args.token_chunk]).to(device)
                real_mask = topk_mask(P_logged, K)  # (t, i, E)
                del P_logged

            for N in range(1, max_N + 1):
                tgt = torch.arange(N + 1, L, device=device)  # i >= N+1
                if tgt.numel() == 0:
                    continue
                predN = tk[:, tgt, tgt - N, :]  # (t, n, E)
                predN1 = tk[:, tgt, tgt - (N + 1), :]  # (t, n, E)
                S = predN | predN1  # union of the two predicted sets
                R = real_mask[:, tgt, :]
                inter = (S & R).sum(-1)  # (t, n)
                union = (S | R).sum(-1)
                iou = inter.float() / union.float()
                superset = (inter == K)  # all real experts covered by S

                tgt_np = tgt.cpu().numpy()
                sum_superset[tgt_np, N] += superset.double().sum(0).cpu().numpy()
                sum_iou[tgt_np, N] += iou.double().sum(0).cpu().numpy()
                sumsq_iou[tgt_np, N] += (iou.double() ** 2).sum(0).cpu().numpy()
                count[tgt_np, N] += t

            # single-offset baseline: prediction = top-k of one offset M
            for M in range(1, max_M_single + 1):
                tgt = torch.arange(M, L, device=device)  # i >= M
                predM = tk[:, tgt, tgt - M, :]  # (t, n, E)
                R = real_mask[:, tgt, :]
                inter = (predM & R).sum(-1)  # (t, n)
                superset = (inter == K)  # single offset covers the real top-k
                tgt_np = tgt.cpu().numpy()
                sum_superset_single[tgt_np, M] += superset.double().sum(0).cpu().numpy()
                count_single[tgt_np, M] += t

            del probs, tk, real_mask, G
        del G_np, P_np
        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"[{fi + 1}/{len(files)}] tokens so far: {int(count.max())}")

    # ----- aggregate -----
    with np.errstate(invalid="ignore", divide="ignore"):
        superset_rate = sum_superset / count
        mean_iou = sum_iou / count
        var_iou = sumsq_iou / count - mean_iou**2
        std_iou = np.sqrt(np.clip(var_iou, 0, None))
    superset_rate[count == 0] = np.nan
    mean_iou[count == 0] = np.nan
    std_iou[count == 0] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        superset_rate_single = sum_superset_single / count_single
    superset_rate_single[count_single == 0] = np.nan
    marg_superset_single = np.full(L, np.nan)
    for M in range(1, max_M_single + 1):
        tot = count_single[:, M].sum()
        if tot > 0:
            marg_superset_single[M] = sum_superset_single[:, M].sum() / tot

    # marginal over layers (function of N), token-weighted
    marg_superset = np.full(L, np.nan)
    marg_iou = np.full(L, np.nan)
    for N in range(1, max_N + 1):
        c = count[:, N]
        tot = c.sum()
        if tot > 0:
            marg_superset[N] = sum_superset[:, N].sum() / tot
            marg_iou[N] = sum_iou[:, N].sum() / tot

    # ----- save -----
    np.savez(
        out_dir / "union_offset_metrics.npz",
        superset_rate=superset_rate,
        mean_iou=mean_iou,
        std_iou=std_iou,
        count=count,
        marg_superset=marg_superset,
        marg_iou=marg_iou,
        superset_rate_single=superset_rate_single,
        count_single=count_single,
        marg_superset_single=marg_superset_single,
        topk=K,
        num_layers=L,
    )
    with open(out_dir / "union_offset_by_layer_N.csv", "w") as f:
        f.write("layer,N,superset_rate,mean_iou,std_iou,count\n")
        for i in range(L):
            for N in range(1, max_N + 1):
                if count[i, N] == 0:
                    continue
                f.write(
                    f"{i},{N},{superset_rate[i, N]:.6g},{mean_iou[i, N]:.6g},"
                    f"{std_iou[i, N]:.6g},{count[i, N]}\n"
                )
    with open(out_dir / "union_offset_marginal_by_N.csv", "w") as f:
        f.write("N,superset_rate,mean_iou\n")
        for N in range(1, max_N + 1):
            f.write(f"{N},{marg_superset[N]:.6g},{marg_iou[N]:.6g}\n")
    with open(out_dir / "single_offset_superset_marginal_by_M.csv", "w") as f:
        f.write("M,superset_rate_single\n")
        for M in range(1, max_M_single + 1):
            f.write(f"{M},{marg_superset_single[M]:.6g}\n")

    _make_figure(out_dir, marg_superset, marg_iou, superset_rate, mean_iou, max_N, K)
    _make_single_superset_heatmap(out_dir, superset_rate_single, max_M_single, K)

    print("\nHEADLINE (averaged over layers):")
    for N in (1, 2, max_N // 2, max_N):
        print(
            f"  N={N:2d}: union superset={marg_superset[N]:.4f}  "
            f"mean_IoU={marg_iou[N]:.4f}  ||  "
            f"single-offset superset@M={N}: {marg_superset_single[N]:.4f}"
        )
    print(f"\noutputs written to {out_dir}")


def _make_single_superset_heatmap(out_dir, superset_rate_single, max_M, K) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib unavailable, skipping single-offset heatmap: {e}")
        return

    L = superset_rate_single.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        superset_rate_single[:, 1 : max_M + 1],
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1,
        extent=[0.5, max_M + 0.5, -0.5, L - 0.5],
    )
    ax.set_xlabel("offset M (single offset)")
    ax.set_ylabel("target layer i")
    ax.set_title(
        f"single-offset superset rate: P[ top-{K}@M ⊇ real top-{K} ] "
        f"(== exact top-{K} match)"
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_superset_rate_single.png", dpi=130)
    plt.close(fig)


def _make_figure(out_dir, marg_superset, marg_iou, superset_rate, mean_iou,
                 max_N, K) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"matplotlib unavailable, skipping figure: {e}")
        return

    Ns = np.arange(1, max_N + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(Ns, marg_superset[1 : max_N + 1], "o-", color="tab:green")
    ax1.set_xlabel("offset N (combining offsets N and N+1)")
    ax1.set_ylabel(f"P[ union(top-{K}@N, top-{K}@N+1) ⊇ real top-{K} ]")
    ax1.set_title("Superset rate: do N and N+1 together cover the real top-k?")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(Ns, marg_iou[1 : max_N + 1], "s-", color="tab:purple")
    ax2.set_xlabel("offset N (combining offsets N and N+1)")
    ax2.set_ylabel(f"mean IoU( union(N, N+1), real top-{K} )")
    ax2.set_title("IoU of combined {N, N+1} predictions with real top-k")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Combined offsets N & N+1 vs real top-{K} routing "
        "(averaged over layers & decode tokens)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "union_offset_two_graphs.png", dpi=140)
    plt.close(fig)

    # bonus: per-(layer, N) heatmaps
    for arr, name, title in [
        (superset_rate, "heatmap_superset_rate.png", "superset rate"),
        (mean_iou, "heatmap_iou.png", "mean IoU"),
    ]:
        L = arr.shape[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(arr[:, 1 : max_N + 1], aspect="auto", origin="lower",
                       vmin=0, vmax=1, extent=[0.5, max_N + 0.5, -0.5, L - 0.5])
        ax.set_xlabel("offset N")
        ax.set_ylabel("target layer i")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
