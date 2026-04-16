#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
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


def compute_instance_flattened_load(instance_dir: Path) -> list[int]:
    temporal_dir = instance_dir / "temporal_expert_logs"
    chunk_files = sorted(
        temporal_dir.glob("*.msgpack.zlib"),
        key=lambda p: (
            _extract_session_timestamp(p),
            _extract_chunk_index(p),
            p.name,
        ),
    )

    flattened_loads: list[int] = []

    for chunk_file in chunk_files:
        payload = _decode_temporal_chunk(chunk_file)
        for iteration in payload.get("iterations", []):
            layers = iteration.get("layers", [])
            iteration_layer_loads = [_layer_request_load(layer) for layer in layers]
            flattened_loads.extend(iteration_layer_loads)

    return flattened_loads


def plot_decode_temporal_load(
    logs_root: Path,
    output_path: Path,
    show: bool,
) -> None:
    decode_instances = discover_decode_instances(logs_root)
    if not decode_instances:
        raise FileNotFoundError(
            f"No decode instance directories with temporal logs found under: {logs_root}"
        )

    plt.figure(figsize=(12, 6))
    any_data = False

    for instance_dir in decode_instances:
        flattened_loads = compute_instance_flattened_load(instance_dir)
        if not flattened_loads:
            continue

        any_data = True
        plt.plot(flattened_loads, linewidth=1.7, label=instance_dir.name)

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
    print(f"Decode instances plotted: {len(decode_instances)}")

    if show:
        plt.show()
    else:
        plt.close()


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

    plot_decode_temporal_load(
        logs_root=logs_root,
        output_path=output_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
