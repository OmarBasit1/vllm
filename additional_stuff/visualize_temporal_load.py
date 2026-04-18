#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import msgspec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot flattened decode temporal MoE load across iterations for all "
            "decode instances in a logs folder."
        )
    )
    parser.add_argument(
        "logs_root",
        type=Path,
        help=(
            "Root folder containing decode instance directories (e.g. "
            "qwen_disagg_example_2gpu)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: <logs_root>/decode_temporal_load.png)."
        ),
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help=(
            "Output interactive HTML path "
            "(default: <logs_root>/decode_temporal_load.html)."
        ),
    )
    parser.add_argument(
        "--cdf-output",
        type=Path,
        default=None,
        help=(
            "Output CDF PNG path (default: <logs_root>/decode_load_spread_cdf.png)."
        ),
    )
    parser.add_argument(
        "--iteration-scatter-dir",
        "--layer-scatter-dir",
        dest="iteration_scatter_dir",
        type=Path,
        default=None,
        help=(
            "Directory for iteration load-vs-latency scatter artifacts "
            "(default: <logs_root>/decode_iteration_load_latency)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of threads to use for chunk decode/parsing (default: 4).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    return parser.parse_args()


def discover_decode_instances(logs_root: Path) -> list[Path]:
    instances: list[Path] = []
    for child in sorted(logs_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("decode_"):
            continue
        temporal_dir = child / "temporal_expert_logs"
        if not temporal_dir.is_dir():
            continue
        if not any(temporal_dir.glob("*.msgpack.zlib")):
            continue
        instances.append(child)
    return instances


def _extract_chunk_index(path: Path) -> int:
    name = path.name
    marker = "_chunk"
    idx = name.rfind(marker)
    if idx < 0:
        return -1
    chunk_str = name[idx + len(marker) :].split(".", maxsplit=1)[0]
    try:
        return int(chunk_str)
    except ValueError:
        return -1


def _extract_session_timestamp(path: Path) -> str:
    name = path.name
    marker = "_ts"
    idx = name.rfind(marker)
    if idx < 0:
        return ""
    return name[idx + len(marker) :].split("_chunk", maxsplit=1)[0]


def _extract_rank_key(path: Path) -> str:
    for token in path.name.split("_"):
        if token.startswith("ep") and token[2:].isdigit():
            return token
        if token.startswith("tp") and token[2:].isdigit():
            return token
    return "tp0"


def _decode_temporal_chunk(path: Path) -> dict[str, Any]:
    compressed = path.read_bytes()
    return msgspec.msgpack.decode(zlib.decompress(compressed))


def _layer_request_load(layer_record: dict[str, Any]) -> int:
    """
    Load for one layer in one iteration is:
    number of requests that activated at least one expert in that layer.
    """
    request_expert_ids = layer_record.get("request_expert_ids", [])
    layer_load = 0
    for request_tokens in request_expert_ids:
        has_activation = any(len(token_experts) > 0 for token_experts in request_tokens)
        if has_activation:
            layer_load += 1
    return layer_load


def _iteration_latency_ms(iteration_record: dict[str, Any]) -> float | None:
    raw_latency = iteration_record.get("iteration_time_ms")
    if raw_latency is None:
        return None

    try:
        latency_ms = float(raw_latency)
    except (TypeError, ValueError):
        return None

    if latency_ms < 0:
        return None

    return latency_ms


def _iteration_start_load(iteration_record: dict[str, Any]) -> int:
    request_token_counts = iteration_record.get("request_token_counts", [])
    start_load = 0
    for count in request_token_counts:
        try:
            if int(count) > 0:
                start_load += 1
        except (TypeError, ValueError):
            continue
    return start_load


def _decode_chunk_payloads(
    chunk_files: list[Path],
    workers: int,
) -> list[dict[str, Any]]:
    if not chunk_files:
        return []
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_decode_temporal_chunk, chunk_files))


def compute_instance_load_metrics(
    instance_dir: Path,
    workers: int,
) -> tuple[
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[tuple[int, float]]],
]:
    temporal_dir = instance_dir / "temporal_expert_logs"
    files_by_rank: dict[str, list[Path]] = defaultdict(list)
    for chunk_file in temporal_dir.glob("*.msgpack.zlib"):
        files_by_rank[_extract_rank_key(chunk_file)].append(chunk_file)

    flattened_loads_by_rank: dict[str, list[int]] = {}
    spread_diffs_by_rank: dict[str, list[int]] = {}
    iteration_latency_points_by_rank: dict[str, list[tuple[int, float]]] = defaultdict(
        list
    )
    for rank_key in sorted(files_by_rank):
        flattened_loads: list[int] = []
        spread_diffs: list[int] = []
        chunk_files = sorted(
            files_by_rank[rank_key],
            key=lambda p: (
                _extract_session_timestamp(p),
                _extract_chunk_index(p),
                p.name,
            ),
        )
        chunk_payloads = _decode_chunk_payloads(chunk_files, workers=workers)
        for payload in chunk_payloads:
            for iteration in payload.get("iterations", []):
                layers = iteration.get("layers", [])
                iteration_layer_loads: list[int] = []
                for layer in layers:
                    layer_load = _layer_request_load(layer)
                    iteration_layer_loads.append(layer_load)

                iteration_latency_ms = _iteration_latency_ms(iteration)
                if iteration_latency_ms is not None:
                    start_load = _iteration_start_load(iteration)
                    if start_load > 0:
                        iteration_latency_points_by_rank[rank_key].append(
                            (start_load, iteration_latency_ms)
                        )

                if not iteration_layer_loads:
                    continue

                flattened_loads.extend(iteration_layer_loads)
                spread_diffs.append(
                    max(iteration_layer_loads) - min(iteration_layer_loads)
                )
        if flattened_loads:
            flattened_loads_by_rank[rank_key] = flattened_loads
        if spread_diffs:
            spread_diffs_by_rank[rank_key] = spread_diffs

    iteration_latency_points_by_rank_dict = {
        rank_key: points
        for rank_key, points in sorted(iteration_latency_points_by_rank.items())
    }

    return (
        flattened_loads_by_rank,
        spread_diffs_by_rank,
        iteration_latency_points_by_rank_dict,
    )


def _build_cdf(values: list[int]) -> tuple[list[int], list[float]]:
    if not values:
        return [], []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cdf = [(idx + 1) / n for idx in range(n)]
    return sorted_vals, cdf


def plot_decode_temporal_load(
    logs_root: Path,
    output_path: Path,
    cdf_output_path: Path,
    html_output_path: Path,
    iteration_scatter_dir: Path,
    show: bool,
    workers: int,
) -> None:
    decode_instances = discover_decode_instances(logs_root)
    if not decode_instances:
        raise FileNotFoundError(
            "No decode instance directories with temporal logs found under: "
            f"{logs_root}"
        )

    plt.figure(figsize=(12, 6))
    any_data = False
    plot_series: list[tuple[str, list[int]]] = []
    cdf_series: list[tuple[str, list[int], list[float]]] = []
    iteration_scatter_data: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for instance_dir in decode_instances:
        (
            flattened_loads_by_rank,
            spread_diffs_by_rank,
            iteration_latency_points_by_rank,
        ) = compute_instance_load_metrics(instance_dir, workers=workers)
        if not flattened_loads_by_rank:
            continue

        for rank_key, flattened_loads in flattened_loads_by_rank.items():
            any_data = True
            series_name = f"{instance_dir.name}:{rank_key}"
            plot_series.append((series_name, flattened_loads))

            spread_diffs = spread_diffs_by_rank.get(rank_key, [])
            cdf_x, cdf_y = _build_cdf(spread_diffs)
            if cdf_x:
                cdf_series.append((series_name, cdf_x, cdf_y))

            plt.plot(
                flattened_loads,
                linewidth=1.7,
                label=series_name,
            )

            scatter_points = iteration_latency_points_by_rank.get(rank_key, [])
            if scatter_points:
                iteration_scatter_data[series_name].extend(scatter_points)

    if not any_data:
        raise RuntimeError(
            "Temporal log files were found, but no iteration data was parsed."
        )

    plt.title("Decode Temporal MoE Load Across Iterations (Layers Flattened)")
    plt.xlabel("Flattened iteration-layer index")
    plt.ylabel("Load")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot: {output_path}")

    if not cdf_series:
        raise RuntimeError("No load spread data available to build CDF plot.")

    plt.figure(figsize=(12, 6))
    for series_name, cdf_x, cdf_y in cdf_series:
        plt.plot(cdf_x, cdf_y, linewidth=1.7, label=series_name)
    plt.title("CDF of Decode Iteration Load Spread (max(layer_load) - min(layer_load))")
    plt.xlabel("Iteration load spread")
    plt.ylabel("CDF")
    plt.xlim(left=0)
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    cdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cdf_output_path, dpi=180)
    print(f"Saved CDF plot: {cdf_output_path}")

    html_output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_interactive_html(
        html_output_path,
        plot_series=plot_series,
        cdf_series=cdf_series,
    )
    print(f"Saved interactive HTML: {html_output_path}")

    _write_iteration_load_latency_plot(
        iteration_scatter_data=iteration_scatter_data,
        output_dir=iteration_scatter_dir,
    )

    print(f"Decode instances plotted: {len(decode_instances)}")

    if show:
        plt.show()
    else:
        plt.close("all")


def _write_interactive_html(
    html_output_path: Path,
    plot_series: list[tuple[str, list[int]]],
    cdf_series: list[tuple[str, list[int], list[float]]],
) -> None:
    traces: list[dict[str, Any]] = []
    for series_name, flattened_loads in plot_series:
        traces.append(
            {
                "x": list(range(len(flattened_loads))),
                "y": flattened_loads,
                "mode": "lines",
                "type": "scatter",
                "name": series_name,
                "line": {"width": 2},
            }
        )

    cdf_traces: list[dict[str, Any]] = []
    for series_name, cdf_x, cdf_y in cdf_series:
        cdf_traces.append(
            {
                "x": cdf_x,
                "y": cdf_y,
                "mode": "lines",
                "type": "scatter",
                "name": series_name,
                "line": {"width": 2},
            }
        )

    line_layout = {
        "title": "Decode Temporal MoE Load Across Iterations (Layers Flattened)",
        "xaxis": {"title": "Flattened iteration-layer index"},
        "yaxis": {"title": "Load"},
        "hovermode": "closest",
        "legend": {"orientation": "v"},
        "template": "plotly_white",
    }

    cdf_layout = {
        "title": "CDF of Decode Iteration Load Spread (max-min)",
        "xaxis": {"title": "Iteration load spread"},
        "yaxis": {"title": "CDF", "range": [0, 1.01]},
        "hovermode": "closest",
        "legend": {"orientation": "v"},
        "template": "plotly_white",
    }

    html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Decode Temporal MoE Load</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
</head>
<body>
  <div id=\"decode-temporal-load\" style=\"width:100%;height:70vh;\"></div>
  <div id=\"decode-load-spread-cdf\" style=\"width:100%;height:70vh;\"></div>
  <script>
    const traces = __TRACES__;
    const cdfTraces = __CDF_TRACES__;
    const lineLayout = __LINE_LAYOUT__;
    const cdfLayout = __CDF_LAYOUT__;
    const config = {
      responsive: true,
      displaylogo: false,
      scrollZoom: true
    };
    Plotly.newPlot('decode-temporal-load', traces, lineLayout, config);
    Plotly.newPlot('decode-load-spread-cdf', cdfTraces, cdfLayout, config);
  </script>
</body>
</html>
"""
    html = html.replace("__TRACES__", json.dumps(traces))
    html = html.replace("__CDF_TRACES__", json.dumps(cdf_traces))
    html = html.replace("__LINE_LAYOUT__", json.dumps(line_layout))
    html = html.replace("__CDF_LAYOUT__", json.dumps(cdf_layout))
    html_output_path.write_text(html, encoding="utf-8")


def _write_iteration_load_latency_plot(
    iteration_scatter_data: dict[str, list[tuple[int, float]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "load_latency_points.csv"
    csv_row_count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x_load", "y_latency_ms"])

        for series_name in sorted(iteration_scatter_data):
            points = iteration_scatter_data[series_name]
            for x_load, y_latency_ms in points:
                writer.writerow([x_load, y_latency_ms])
                csv_row_count += 1

    print(f"Saved {csv_row_count} rows to CSV: {csv_path}")

    if csv_row_count == 0:
        print(
            "No per-iteration latency data found (iteration_time_ms missing). "
            f"No scatter plot was written to: {output_dir}"
        )
        return

    plt.figure(figsize=(10, 6))
    for series_name in sorted(iteration_scatter_data):
        points = iteration_scatter_data[series_name]
        if not points:
            continue

        x_load = [load for load, _ in points]
        y_latency_ms = [latency_ms for _, latency_ms in points]
        plt.scatter(
            x_load,
            y_latency_ms,
            s=16,
            alpha=0.65,
            label=series_name,
        )

    plt.title("Iteration Start Load vs Iteration Latency")
    plt.xlabel("Load at iteration start")
    plt.ylabel("Iteration latency (ms)")
    plt.ylim((0, 200))
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path = output_dir / "iteration_load_vs_latency.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved iteration load-vs-latency plot: {output_path}")


def main() -> None:
    args = parse_args()
    logs_root = args.logs_root.expanduser().resolve()
    if not logs_root.is_dir():
        raise NotADirectoryError(f"logs_root is not a directory: {logs_root}")

    output_path = args.output
    if output_path is None:
        output_path = logs_root / "decode_temporal_load.png"
    else:
        output_path = output_path.expanduser().resolve()

    html_output_path = args.html_output
    if html_output_path is None:
        html_output_path = logs_root / "decode_temporal_load.html"
    else:
        html_output_path = html_output_path.expanduser().resolve()

    cdf_output_path = args.cdf_output
    if cdf_output_path is None:
        cdf_output_path = logs_root / "decode_load_spread_cdf.png"
    else:
        cdf_output_path = cdf_output_path.expanduser().resolve()

    iteration_scatter_dir = args.iteration_scatter_dir
    if iteration_scatter_dir is None:
        iteration_scatter_dir = logs_root / "decode_iteration_load_latency"
    else:
        iteration_scatter_dir = iteration_scatter_dir.expanduser().resolve()

    plot_decode_temporal_load(
        logs_root=logs_root,
        output_path=output_path,
        cdf_output_path=cdf_output_path,
        html_output_path=html_output_path,
        iteration_scatter_dir=iteration_scatter_dir,
        show=args.show,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
