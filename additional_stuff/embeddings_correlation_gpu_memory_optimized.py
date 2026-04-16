#!/usr/bin/env python3
"""GPU-first, memory-optimized correlation analysis for MoE decode traces.

This script reads decode rows (split_id=1) from SQLite and computes:
1) cosine similarity of semantic embeddings (layer0_input_embeddings),
2) per-layer Jaccard similarity over expert activation sets (layer_token_topk),
3) per-layer Spearman correlation between cosine and Jaccard.

Memory strategy:
- Stream full pair traversal per layer on GPU.
- Filter out pair datapoints with cosine similarity < 0.5.
- Process Jaccard + Spearman one layer at a time to avoid OOM.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import sqlite3
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DECODE_SPLIT_ID = 1
MIN_COSINE_SIM = 0.9
TOP_K = 12
RANDOM_SAMPLE_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU memory-optimized correlation between semantic embedding cosine "
            "and per-layer expert-set Jaccard (Spearman per layer)."
        )
    )
    parser.add_argument(
        "--sqlite-path",
        default="/export2/obasit/ClusterMoE/vllm-MoE/additional_stuff/artifacts/qwen.sqlite",
        help="Path to SQLite DB with layer0_input_embeddings and layer_token_topk.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/export2/obasit/ClusterMoE/vllm-MoE/additional_stuff/artifacts/correlation"
        ),
        help="Directory for output CSV and plot.",
    )
    parser.add_argument(
        "--output-prefix",
        default="embeddings_expert_jaccard_spearman_decode_gpu_optimized",
        help="Output prefix for CSV and PNG files.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help=(
            "Optional random sample size after loading all eligible decode "
            "points into memory."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device to run on (example: cuda:0, cuda:1).",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=None,
        help="Optional minimum layer id to process (inclusive).",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="Optional maximum layer id to process (inclusive).",
    )
    parser.add_argument(
        "--embedding-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Embedding tensor dtype on GPU.",
    )
    parser.add_argument(
        "--cosine-row-block",
        type=int,
        default=90000,
        help=(
            "Number of request rows per cosine step. Each step compares these "
            "rows against the full embedding matrix."
        ),
    )
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        default=40_000_000,
        help="Number of filtered pairs per chunk for per-layer Jaccard/Spearman.",
    )
    parser.add_argument(
        "--log-every-blocks",
        type=int,
        default=20,
        help="Log progress every N cosine blocks.",
    )
    parser.add_argument(
        "--log-every-pair-chunks",
        type=int,
        default=10000,
        help="Log progress every N pair chunks in per-layer pass.",
    )
    parser.add_argument(
        "--log-every-layer-rows",
        type=int,
        default=1_000_000,
        help="Log progress every N rows while loading one layer's expert sets.",
    )
    parser.add_argument(
        "--future-comparison-iterations",
        type=int,
        default=3,
        help=(
            "Number of iteration-offset comparisons to run. "
            "1 means only i vs i; 3 means i vs i, i vs i+1, i vs i+2."
        ),
    )
    return parser.parse_args()


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("emb_corr_gpu")


def _dtype_from_name(name: str) -> np.dtype:
    lowered = name.strip().lower()
    if lowered in ("float16", "fp16", "half"):
        return np.float16
    if lowered in ("float32", "fp32"):
        return np.float32
    raise ValueError(f"Unsupported embedding dtype in DB: {name}")


def _torch_dtype_from_arg(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype arg: {name}")


def _gpu_mem_gb(device: str) -> tuple[float, float]:
    import torch

    free_b, total_b = torch.cuda.mem_get_info(device=device)
    return free_b / (1024**3), total_b / (1024**3)


def _load_decode_keys_and_layers(
    conn: sqlite3.Connection, logger: logging.Logger
) -> tuple[set[tuple[str, int]], list[int], int]:
    logger.info("Loading decode key set (request_id, iter_no) from layer_token_topk...")
    expert_keys: set[tuple[str, int]] = set()

    cur = conn.execute(
        """
        SELECT DISTINCT request_id, iter_no
        FROM layer_token_topk
        WHERE split_id = ?
        """,
        (DECODE_SPLIT_ID,),
    )
    while True:
        batch = cur.fetchmany(200_000)
        if not batch:
            break
        for request_id, iter_no in batch:
            expert_keys.add((str(request_id), int(iter_no)))
        logger.info("Collected %d decode keys so far...", len(expert_keys))

    layer_rows = conn.execute(
        """
        SELECT DISTINCT layer_no
        FROM layer_token_topk
        WHERE split_id = ?
        ORDER BY layer_no ASC
        """,
        (DECODE_SPLIT_ID,),
    ).fetchall()
    layers = [int(row[0]) for row in layer_rows]

    max_expert_row = conn.execute(
        """
        SELECT MAX(expert_id)
        FROM layer_token_topk
        WHERE split_id = ?
        """,
        (DECODE_SPLIT_ID,),
    ).fetchone()
    max_expert_id = int(max_expert_row[0]) if max_expert_row and max_expert_row[0] is not None else -1

    logger.info(
        "Decode metadata: %d keys, %d layers, max_expert_id=%d",
        len(expert_keys),
        len(layers),
        max_expert_id,
    )
    return expert_keys, layers, max_expert_id


def _load_embeddings(
    conn: sqlite3.Connection,
    *,
    decode_keys_with_experts: set[tuple[str, int]],
    max_points: int | None,
    logger: logging.Logger,
) -> tuple[list[tuple[str, int]], np.ndarray]:
    logger.info("Streaming decode embeddings from layer0_input_embeddings...")

    keys: list[tuple[str, int]] = []
    vectors: list[np.ndarray] = []

    cur = conn.execute(
        """
        SELECT request_id, iter_no, num_tokens, hidden_size, dtype, zlib_blob
        FROM layer0_input_embeddings
        WHERE split_id = ?
        ORDER BY request_id ASC, iter_no ASC
        """,
        (DECODE_SPLIT_ID,),
    )

    scanned = 0
    kept = 0
    while True:
        batch = cur.fetchmany(5_000)
        if not batch:
            break

        for request_id, iter_no, num_tokens, hidden_size, dtype_name, zblob in batch:
            scanned += 1
            key = (str(request_id), int(iter_no))

            if key not in decode_keys_with_experts:
                continue
            if zblob is None or int(num_tokens) <= 0 or int(hidden_size) <= 0:
                continue
            if int(num_tokens) != 1:
                continue

            raw = zlib.decompress(zblob)
            dtype = _dtype_from_name(str(dtype_name))
            arr = np.frombuffer(raw, dtype=dtype)
            expected = int(num_tokens) * int(hidden_size)
            if arr.size != expected:
                continue

            emb = arr.reshape((int(num_tokens), int(hidden_size)))[0].astype(
                np.float32, copy=False
            )
            keys.append(key)
            vectors.append(emb)
            kept += 1

        logger.info("Embeddings scanned=%d kept=%d", scanned, kept)

    if kept < 2:
        raise RuntimeError(
            "Need at least two decode points with both embeddings and expert activations."
        )

    if max_points is not None:
        target = int(max_points)
        if target <= 0:
            raise ValueError("max_points must be > 0 when provided")

        if kept > target:
            logger.info(
                "Applying random subsampling with seed=%d: selecting %d of %d points.",
                RANDOM_SAMPLE_SEED,
                target,
                kept,
            )
            rng = np.random.default_rng(RANDOM_SAMPLE_SEED)
            sample_idx = np.sort(rng.choice(kept, size=target, replace=False))

            sampled_keys = [keys[i] for i in sample_idx.tolist()]
            sampled_vectors = [vectors[i] for i in sample_idx.tolist()]

            # Drop non-selected data references so memory can be reclaimed.
            del keys
            del vectors
            del sample_idx

            keys = sampled_keys
            vectors = sampled_vectors
            kept = target
            logger.info("Subsample complete. kept=%d", kept)
        else:
            logger.info(
                "Requested max_points=%d but only %d points are available; keeping all.",
                target,
                kept,
            )

    emb_matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    del vectors
    logger.info("Final embedding matrix shape: %s", tuple(emb_matrix.shape))
    return keys, emb_matrix


def _popcount_i64_torch(values):
    import torch

    if hasattr(torch, "bitwise_count"):
        return torch.bitwise_count(values)

    # Compatibility fallback when bitwise_count is unavailable.
    work = values.clone()
    counts = torch.zeros_like(work, dtype=torch.int16)
    for _ in range(64):
        nonzero = work != 0
        if not bool(torch.any(nonzero)):
            break
        counts = counts + nonzero.to(dtype=torch.int16)
        work = torch.bitwise_and(work, work - 1)
    return counts


def _load_layer_masks(
    conn: sqlite3.Connection,
    *,
    layer_no: int,
    key_to_idx: dict[tuple[str, int], int],
    num_points: int,
    num_words: int,
    log_every_rows: int,
    logger: logging.Logger,
) -> np.ndarray:
    masks = np.zeros((num_points, num_words), dtype=np.uint64)

    logger.info("Loading expert rows for layer %d...", layer_no)
    cur = conn.execute(
        """
        WITH ranked AS (
            SELECT
                request_id,
                iter_no,
                expert_id,
                ROW_NUMBER() OVER (
                    PARTITION BY split_id, request_id, iter_no, layer_no
                    ORDER BY probability DESC, expert_id ASC
                ) AS prob_rank
            FROM layer_token_topk
            WHERE split_id = ? AND layer_no = ?
        )
        SELECT request_id, iter_no, expert_id
        FROM ranked
        WHERE prob_rank <= ?
        ORDER BY request_id ASC, iter_no ASC
        """,
        (DECODE_SPLIT_ID, layer_no, TOP_K),
    )

    scanned = 0
    applied = 0
    while True:
        batch = cur.fetchmany(500_000)
        if not batch:
            break
        for request_id, iter_no, expert_id in batch:
            scanned += 1
            idx = key_to_idx.get((str(request_id), int(iter_no)))
            if idx is None:
                continue

            expert = int(expert_id)
            if expert < 0:
                continue
            word = expert // 64
            if word >= num_words:
                continue
            bit = expert % 64
            masks[idx, word] |= np.uint64(1) << np.uint64(bit)
            applied += 1

        if scanned % max(1, log_every_rows) == 0:
            logger.info(
                "Layer %d mask load progress: scanned=%d applied=%d",
                layer_no,
                scanned,
                applied,
            )

    logger.info(
        "Layer %d masks ready: scanned=%d applied=%d nonempty_points=%d",
        layer_no,
        scanned,
        applied,
        int(np.count_nonzero(np.any(masks != 0, axis=1))),
    )
    return masks


def _compute_layer_spearman(
    *,
    layer_no: int,
    emb_t,
    layer_masks: np.ndarray,
    cosine_row_block: int,
    pair_chunk_size: int,
    log_every_blocks: int,
    log_every_pair_chunks: int,
    device: str,
    logger: logging.Logger,
    layer_plot_output_path: Path | None = None,
    comparison_offset: int = 0,
) -> tuple[float, float, int, int, int]:
    import torch

    metric_mod = importlib.import_module("torchmetrics.regression")
    SpearmanCorrCoef = getattr(metric_mod, "SpearmanCorrCoef")
    PearsonCorrCoef = getattr(metric_mod, "PearsonCorrCoef")
    metric = SpearmanCorrCoef().to(device=device)
    pearson_metric = PearsonCorrCoef().to(device=device)

    whisker_fig = None
    whisker_ax = None
    jacc_bin_edges = None
    cos_sum_by_bin = None
    cos_sq_sum_by_bin = None
    count_by_bin = None
    if layer_plot_output_path is not None:
        layer_plot_output_path.parent.mkdir(parents=True, exist_ok=True)
        whisker_fig, whisker_ax = plt.subplots(figsize=(7.0, 5.8))
        num_bins = 10
        jacc_bin_edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float32)
        cos_sum_by_bin = np.zeros(num_bins, dtype=np.float64)
        cos_sq_sum_by_bin = np.zeros(num_bins, dtype=np.float64)
        count_by_bin = np.zeros(num_bins, dtype=np.int64)

    def _save_layer_whisker_plot() -> None:
        if (
            whisker_fig is None
            or whisker_ax is None
            or layer_plot_output_path is None
            or jacc_bin_edges is None
            or cos_sum_by_bin is None
            or cos_sq_sum_by_bin is None
            or count_by_bin is None
        ):
            return

        valid_bins = count_by_bin > 0
        centers = 0.5 * (jacc_bin_edges[:-1] + jacc_bin_edges[1:])
        mean_cosine = np.divide(
            cos_sum_by_bin,
            count_by_bin,
            out=np.zeros_like(cos_sum_by_bin),
            where=valid_bins,
        )
        mean_sq = np.divide(
            cos_sq_sum_by_bin,
            count_by_bin,
            out=np.zeros_like(cos_sq_sum_by_bin),
            where=valid_bins,
        )
        std_cosine = np.sqrt(np.maximum(0.0, mean_sq - (mean_cosine * mean_cosine)))

        whisker_ax.errorbar(
            centers[valid_bins],
            mean_cosine[valid_bins],
            yerr=std_cosine[valid_bins],
            fmt="none",
            ecolor="#2E6F95",
            elinewidth=0.9,
            capsize=1.5,
            alpha=0.85,
        )
        whisker_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        whisker_ax.set_xlabel("Jaccard similarity (binned)")
        whisker_ax.set_ylabel("Mean cosine similarity (whiskers=std)")
        whisker_ax.set_xlim(0.0, 1.0)
        whisker_ax.set_ylim(MIN_COSINE_SIM, 1.0)
        whisker_ax.set_title(
            "Layer "
            f"{layer_no}: Cosine iter i vs Jaccard iter i+{comparison_offset} "
            f"(pairs={used_pairs})"
        )
        whisker_fig.tight_layout()
        whisker_fig.savefig(layer_plot_output_path, dpi=220)
        plt.close(whisker_fig)
        logger.info("Wrote layer whisker plot: %s", layer_plot_output_path)

    n = int(emb_t.shape[0])
    emb_t_t = emb_t.transpose(0, 1)
    layer_masks_t = torch.from_numpy(layer_masks.view(np.int64)).to(
        device=device, dtype=torch.int64
    )

    total_pairs = n * (n - 1) // 2
    if total_pairs < 2:
        logger.warning("Layer %d has fewer than 2 pairs; rho=NaN", layer_no)
        return float("nan"), float("nan"), 0, 0, 0

    row_starts = list(range(0, n, max(1, cosine_row_block)))
    total_blocks = len(row_starts)

    used_pairs = 0
    zero_union_dropped = 0
    low_cos_dropped = 0

    t0 = time.time()
    block_idx = 0
    chunk_idx = 0

    for row_start in row_starts:
        row_end = min(n, row_start + max(1, cosine_row_block))
        sim = torch.matmul(emb_t[row_start:row_end], emb_t_t)
        block_idx += 1

        for local_r in range(row_end - row_start):
            i = row_start + local_r
            j0 = i + 1
            if j0 >= n:
                continue

            for j_start in range(j0, n, max(1, pair_chunk_size)):
                j_end = min(n, j_start + max(1, pair_chunk_size))
                chunk_idx += 1

                cos_t = sim[local_r, j_start:j_end].to(dtype=torch.float32)
                mask_j = layer_masks_t[j_start:j_end]
                mask_i = layer_masks_t[i].expand_as(mask_j)

                inter = torch.bitwise_and(mask_i, mask_j)
                union = torch.bitwise_or(mask_i, mask_j)

                inter_cnt = _popcount_i64_torch(inter).sum(dim=1).to(dtype=torch.float32)
                union_cnt = _popcount_i64_torch(union).sum(dim=1).to(dtype=torch.float32)

                valid_union = union_cnt > 0
                valid_cos = cos_t >= MIN_COSINE_SIM
                valid = valid_union & valid_cos
                valid_count = int(valid.sum().item())
                if valid_count > 0:
                    jacc_t = inter_cnt[valid] / union_cnt[valid]

                    metric.update(cos_t[valid], jacc_t)
                    pearson_metric.update(cos_t[valid], jacc_t)
                    used_pairs += valid_count

                    if whisker_ax is not None:
                        cos_np = cos_t[valid].detach().cpu().numpy()
                        jacc_np = jacc_t.detach().cpu().numpy()
                        bin_idx = np.digitize(jacc_np, jacc_bin_edges, right=False) - 1
                        bin_idx = np.clip(bin_idx, 0, len(cos_sum_by_bin) - 1)
                        np.add.at(cos_sum_by_bin, bin_idx, cos_np)
                        np.add.at(cos_sq_sum_by_bin, bin_idx, cos_np * cos_np)
                        np.add.at(count_by_bin, bin_idx, 1)

                zero_union_dropped += int((~valid_union).sum().item())
                low_cos_dropped += int((valid_union & (~valid_cos)).sum().item())

                if chunk_idx % max(1, log_every_pair_chunks) == 0:
                    free_gb, total_gb = _gpu_mem_gb(device)
                    elapsed = time.time() - t0
                    logger.info(
                        "Layer %d chunk %d | row=%d j=[%d:%d) | used=%d "
                        "dropped_zero_union=%d dropped_low_cos=%d | "
                        "GPU free %.2f/%.2f GB | elapsed %.1fs",
                        layer_no,
                        chunk_idx,
                        i,
                        j_start,
                        j_end,
                        used_pairs,
                        zero_union_dropped,
                        low_cos_dropped,
                        free_gb,
                        total_gb,
                        elapsed,
                    )

                del cos_t
                del mask_j
                del mask_i
                del inter
                del union
                del inter_cnt
                del union_cnt
                del valid

        if block_idx % max(1, log_every_blocks) == 0 or block_idx == total_blocks:
            free_gb, total_gb = _gpu_mem_gb(device)
            elapsed = time.time() - t0
            logger.info(
                "Layer %d row block %d/%d | row=[%d:%d) | used=%d "
                "dropped_zero_union=%d dropped_low_cos=%d | "
                "GPU free %.2f/%.2f GB | elapsed %.1fs",
                layer_no,
                block_idx,
                total_blocks,
                row_start,
                row_end,
                used_pairs,
                zero_union_dropped,
                low_cos_dropped,
                free_gb,
                total_gb,
                elapsed,
            )

        del sim

    if used_pairs < 2:
        _save_layer_whisker_plot()
        logger.warning(
            "Layer %d has fewer than 2 valid pairs after filtering; rho=NaN", layer_no
        )
        return float("nan"), float("nan"), used_pairs, zero_union_dropped, low_cos_dropped

    try:
        rho = float(metric.compute().item())
        pearson_r = float(pearson_metric.compute().item())
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "GPU OOM while computing Spearman. Reduce --max-points or split "
                "work by layers across two GPUs."
            ) from exc
        raise

    _save_layer_whisker_plot()

    return rho, pearson_r, used_pairs, zero_union_dropped, low_cos_dropped


def _write_csv(output_csv: Path, rows: list[dict[str, float | int]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "spearman_rho",
                "pearson_r",
                "pairs_used",
                "pairs_dropped_zero_union",
                "pairs_dropped_low_cosine",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_plot(
    *,
    layers: list[int],
    rho_by_layer: list[float],
    pearson_by_layer: list[float],
    output_path: Path,
    comparison_offset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(layers))
    fig_w = max(10.0, 0.35 * len(layers) + 4.0)

    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 9.0), sharex=True)

    axes[0].bar(x, rho_by_layer, color="#2E6F95", alpha=0.9)
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("Spearman rank correlation (rho)")
    axes[0].set_title(
        "Decode: Cosine iter i vs Expert-Set Jaccard iter "
        f"i+{comparison_offset}"
    )
    axes[0].set_ylim(-1.0, 1.0)

    axes[1].bar(x, pearson_by_layer, color="#3C9D73", alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(l) for l in layers], rotation=90)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Pearson correlation (r)")
    axes[1].set_ylim(-1.0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> tuple[Path, Path]:
    import torch
    import torch.nn.functional as F

    logger = _setup_logger()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    try:
        torch.cuda.get_device_properties(torch.device(args.device))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Invalid CUDA device '{args.device}': {exc}") from exc

    logger.info("Using device: %s", args.device)
    free_gb, total_gb = _gpu_mem_gb(args.device)
    logger.info("Initial GPU memory free %.2f / %.2f GB", free_gb, total_gb)

    num_comparisons = int(args.future_comparison_iterations)
    if num_comparisons <= 0:
        raise ValueError("--future-comparison-iterations must be > 0")

    conn = sqlite3.connect(str(args.sqlite_path))
    try:
        decode_keys, layers, max_expert_id = _load_decode_keys_and_layers(conn, logger)
        if not layers:
            raise RuntimeError("No decode layers found in layer_token_topk.")

        if args.layer_start is not None:
            layers = [layer for layer in layers if layer >= int(args.layer_start)]
        if args.layer_end is not None:
            layers = [layer for layer in layers if layer <= int(args.layer_end)]
        if not layers:
            raise RuntimeError(
                "No layers left after applying --layer-start/--layer-end filters."
            )

        keys, emb_matrix = _load_embeddings(
            conn,
            decode_keys_with_experts=decode_keys,
            max_points=args.max_points,
            logger=logger,
        )

        num_points = int(emb_matrix.shape[0])
        hidden = int(emb_matrix.shape[1])
        logger.info("Analysis points=%d hidden_size=%d layers=%d", num_points, hidden, len(layers))

        key_to_idx = {k: i for i, k in enumerate(keys)}
        num_words = max(1, (max_expert_id // 64) + 1)
        logger.info("Bitmask words per point: %d", num_words)

        emb_dtype = _torch_dtype_from_arg(args.embedding_dtype)
        emb_t = torch.from_numpy(emb_matrix).to(device=args.device, dtype=emb_dtype)
        emb_t = F.normalize(emb_t, p=2.0, dim=1, eps=1e-12)
        logger.info("Uploaded and normalized embeddings on GPU (%s).", args.embedding_dtype)

        logger.info(
            "Using full pair traversal with cosine filter: cosine >= %.2f",
            MIN_COSINE_SIM,
        )

        rows: list[dict[str, float | int]] = []
        out_dir = Path(args.output_dir)
        logger.info(
            "Running %d comparison offsets: 0..%d",
            num_comparisons,
            num_comparisons - 1,
        )

        first_output_csv: Path | None = None
        first_output_plot: Path | None = None

        for comparison_offset in range(num_comparisons):
            rows = []
            rho_by_layer: list[float] = []
            pearson_by_layer: list[float] = []

            comparison_tag = f"offset_{comparison_offset:02d}"
            comparison_dir = out_dir / f"{args.output_prefix}_{comparison_tag}"
            whisker_dir = comparison_dir / f"{args.output_prefix}_whisker_layers"
            whisker_dir.mkdir(parents=True, exist_ok=True)

            shifted_key_to_idx = {
                (request_id, iter_no + comparison_offset): idx
                for idx, (request_id, iter_no) in enumerate(keys)
            }

            logger.info(
                "Starting comparison offset=%d (cosine iter i vs jaccard iter i+%d)",
                comparison_offset,
                comparison_offset,
            )

            for layer in layers:
                logger.info(
                    "Starting per-layer pass for layer %d at offset=%d",
                    layer,
                    comparison_offset,
                )
                layer_masks = _load_layer_masks(
                    conn,
                    layer_no=layer,
                    key_to_idx=shifted_key_to_idx,
                    num_points=num_points,
                    num_words=num_words,
                    log_every_rows=max(1, int(args.log_every_layer_rows)),
                    logger=logger,
                )

                layer_plot_output_path = (
                    whisker_dir
                    / (
                        f"{args.output_prefix}_{comparison_tag}"
                        f"_layer_{int(layer):03d}.png"
                    )
                )

                rho, pearson_r, used_pairs, dropped_pairs, dropped_low_cos = _compute_layer_spearman(
                    layer_no=layer,
                    emb_t=emb_t,
                    layer_masks=layer_masks,
                    cosine_row_block=max(1, int(args.cosine_row_block)),
                    pair_chunk_size=max(1, int(args.pair_chunk_size)),
                    log_every_blocks=max(1, int(args.log_every_blocks)),
                    log_every_pair_chunks=max(1, int(args.log_every_pair_chunks)),
                    device=args.device,
                    logger=logger,
                    layer_plot_output_path=layer_plot_output_path,
                    comparison_offset=comparison_offset,
                )

                rows.append(
                    {
                        "layer": int(layer),
                        "spearman_rho": float(rho),
                        "pearson_r": float(pearson_r),
                        "pairs_used": int(used_pairs),
                        "pairs_dropped_zero_union": int(dropped_pairs),
                        "pairs_dropped_low_cosine": int(dropped_low_cos),
                    }
                )
                rho_by_layer.append(float(rho))
                pearson_by_layer.append(float(pearson_r))

                logger.info(
                    "Layer %d offset=%d complete: spearman=%.6f pearson=%.6f used_pairs=%d dropped_zero_union=%d dropped_low_cos=%d",
                    layer,
                    comparison_offset,
                    rho,
                    pearson_r,
                    used_pairs,
                    dropped_pairs,
                    dropped_low_cos,
                )

            output_csv = comparison_dir / f"{args.output_prefix}_{comparison_tag}.csv"
            output_plot = comparison_dir / f"{args.output_prefix}_{comparison_tag}.png"

            _write_csv(output_csv, rows)
            _build_plot(
                layers=layers,
                rho_by_layer=rho_by_layer,
                pearson_by_layer=pearson_by_layer,
                output_path=output_plot,
                comparison_offset=comparison_offset,
            )

            logger.info("Wrote CSV: %s", output_csv)
            logger.info("Wrote plot: %s", output_plot)
            logger.info("Wrote per-layer whisker plots under: %s", whisker_dir)

            if first_output_csv is None:
                first_output_csv = output_csv
            if first_output_plot is None:
                first_output_plot = output_plot

        del emb_t
        torch.cuda.empty_cache()

        if first_output_csv is None or first_output_plot is None:
            raise RuntimeError("No comparison outputs were generated.")

        return first_output_csv, first_output_plot
    finally:
        conn.close()


def main() -> None:
    args = _parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
