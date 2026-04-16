#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import zlib
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
            "Output interactive HTML path (default: <logs_root>/decode_temporal_load.html)."
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
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    temporal_dir = instance_dir / "temporal_expert_logs"
    files_by_rank: dict[str, list[Path]] = defaultdict(list)
    for chunk_file in temporal_dir.glob("*.msgpack.zlib"):
        files_by_rank[_extract_rank_key(chunk_file)].append(chunk_file)

    flattened_loads_by_rank: dict[str, list[int]] = {}
    spread_diffs_by_rank: dict[str, list[int]] = {}
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
                iteration_layer_loads = [_layer_request_load(layer) for layer in layers]
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

    return flattened_loads_by_rank, spread_diffs_by_rank


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
    show: bool,
    workers: int,
) -> None:
    decode_instances = discover_decode_instances(logs_root)
    if not decode_instances:
        raise FileNotFoundError(
            f"No decode instance directories with temporal logs found under: {logs_root}"
        )

    plt.figure(figsize=(12, 6))
    any_data = False
    plot_series: list[tuple[str, list[int]]] = []
    cdf_series: list[tuple[str, list[int], list[float]]] = []

    for instance_dir in decode_instances:
        flattened_loads_by_rank, spread_diffs_by_rank = compute_instance_load_metrics(
            instance_dir,
            workers=workers,
        )
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

    if not any_data:
        raise RuntimeError("Temporal log files were found, but no iteration data was parsed.")

    plt.title("Decode Temporal MoE Load Across Iterations (Layers Flattened)")
    plt.xlabel("Flattened iteration-layer index")
    plt.ylabel("Load")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot: {output_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(cdf_series[-1][1], cdf_series[-1][2], linewidth=1.7, label=series_name)
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

    plot_decode_temporal_load(
        logs_root=logs_root,
        output_path=output_path,
        cdf_output_path=cdf_output_path,
        html_output_path=html_output_path,
        show=args.show,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
