#!/usr/bin/env python3
"""Generate normalized expert-activation heatmaps from MoE SQLite logs.

The heatmap uses:
- Y-axis: layer id
- X-axis: expert id
- Value: normalized expert activation per layer

For each unique (split_id, request_id, iter_no, layer_no), we keep only the
top-N experts by probability, then aggregate expert counts per layer.

Normalization is done per layer:
    count(layer, expert in top-N) / total_count(layer, all experts in top-N)
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPLIT_TO_ID: dict[str, int] = {"prefill": 0, "decode": 1}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a normalized layer-vs-expert activation heatmap "
            "from a MoE profiling SQLite DB."
        )
    )
    parser.add_argument(
        "--sqlite-path",
        default="/export2/obasit/ClusterMoE/vllm-MoE/additional_stuff/artifacts/qwen.sqlite",
        help="Path to SQLite DB containing layer_token_topk table.",
    )
    parser.add_argument(
        "--output-dir",
        default="./",
        help="Directory to write heatmap image(s).",
    )
    parser.add_argument(
        "--output-name",
        default="normalized_expert_activation_heatmap.png",
        help="Output image filename (default: normalized_expert_activation_heatmap.png).",
    )
    parser.add_argument(
        "--split",
        choices=("prefill", "decode", "both"),
        default="both",
        help="Which split to use for the heatmap (default: both).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=4,
        help=(
            "Per (split_id, request_id, iter_no, layer_no), keep only the "
            "top-N experts by probability (default: 4)."
        ),
    )
    return parser.parse_args()


def _read_counts(
    sqlite_path: str | Path,
    split: str,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        where_clause = ""
        params: tuple[int, ...] = ()
        if split != "both":
            split_id = SPLIT_TO_ID[split]
            where_clause = "WHERE split_id = ?"
            params = (split_id,)

        cursor = conn.execute(
            f"""
            WITH ranked AS (
                SELECT
                    layer_no,
                    expert_id,
                    probability,
                    ROW_NUMBER() OVER (
                        PARTITION BY split_id, request_id, iter_no, layer_no
                        ORDER BY probability DESC, expert_id ASC
                    ) AS prob_rank
                FROM layer_token_topk
                {where_clause}
            )
            SELECT
                layer_no,
                expert_id,
                COUNT(*) AS activation_count
            FROM ranked
            WHERE prob_rank <= ?
            GROUP BY layer_no, expert_id
            ORDER BY layer_no ASC, expert_id ASC
            """,
            (*params, top_n),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No activation rows found in layer_token_topk.")

    data = np.asarray(rows, dtype=np.int64)
    layer_ids = np.unique(data[:, 0])
    expert_ids = np.unique(data[:, 1])

    layer_to_idx = {layer_id: idx for idx, layer_id in enumerate(layer_ids.tolist())}
    expert_to_idx = {
        expert_id: idx for idx, expert_id in enumerate(expert_ids.tolist())
    }

    counts = np.zeros((len(layer_ids), len(expert_ids)), dtype=np.float64)
    for layer_id, expert_id, activation_count in data.tolist():
        counts[layer_to_idx[layer_id], expert_to_idx[expert_id]] = activation_count

    return layer_ids, expert_ids, counts


def _normalize_per_layer(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals > 0,
    )
    return normalized


def _downsample_ticks(ids: np.ndarray, max_ticks: int) -> tuple[np.ndarray, list[str]]:
    if len(ids) <= max_ticks:
        positions = np.arange(len(ids))
        labels = [str(x) for x in ids.tolist()]
        return positions, labels

    step = int(np.ceil(len(ids) / max_ticks))
    positions = np.arange(0, len(ids), step)
    labels = [str(ids[pos]) for pos in positions.tolist()]
    return positions, labels


def generate_heatmap(
    *,
    sqlite_path: str | Path,
    output_dir: str | Path,
    output_name: str,
    split: str,
    top_n: int,
) -> Path:
    layer_ids, expert_ids, counts = _read_counts(sqlite_path, split, top_n)
    normalized = _normalize_per_layer(counts)

    print(
        f"Read top-{top_n} counts for {len(layer_ids)} layers and "
        f"{len(expert_ids)} experts."
    )
    print(f"shape of counts: {counts.shape}, shape of normalized: {normalized.shape}")

    # Assert that each layer sums to 1
    layer_sums = normalized.sum(axis=1)
    assert np.allclose(layer_sums, 1.0), f"Layer sums not close to 1.0: {layer_sums}"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / output_name

    fig_w = max(8.0, 3.0 + 0.22 * len(expert_ids))
    fig_h = max(5.0, 2.5 + 0.22 * len(layer_ids))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    heatmap = ax.imshow(normalized, aspect="auto", cmap="viridis", vmin=0.0)

    x_positions, x_labels = _downsample_ticks(expert_ids, max_ticks=30)
    y_positions, y_labels = _downsample_ticks(layer_ids, max_ticks=30)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer ID")
    ax.set_title(f"Normalized Expert Activation Per Layer ({split}, top-{top_n})")

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Activation fraction within layer")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()
    out_path = generate_heatmap(
        sqlite_path=args.sqlite_path,
        output_dir=args.output_dir,
        output_name=args.output_name,
        split=args.split,
        top_n=args.top_n,
    )
    print(f"Heatmap written to: {out_path}")


if __name__ == "__main__":
    main()
