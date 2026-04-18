#!/usr/bin/env python3
"""Run aiPerf profile with a JSON config file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to aiPerf JSON config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the final command and exit",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cmd(cfg: Dict[str, Any]) -> List[str]:
    required = ["model", "url", "endpoint_type"]
    for key in required:
        if not cfg.get(key):
            raise ValueError(f"Missing required config key: {key}")

    cmd = [
        str(cfg.get("aiperf_binary", "aiperf")),
        "profile",
        "--model",
        str(cfg["model"]),
        "--url",
        str(cfg["url"]),
        "--endpoint-type",
        str(cfg["endpoint_type"]),
    ]

    optional_flags = [
        ("input_file", "--input-file"),
        ("custom_dataset_type", "--custom-dataset-type"),
        ("concurrency", "--concurrency"),
        ("request_rate", "--request-rate"),
        ("arrival_pattern", "--arrival-pattern"),
        ("request_count", "--request-count"),
        ("max_concurrency", "--max-concurrency"),
        ("api_key", "--api-key"),
        ("public_dataset", "--public-dataset"),
    ]
    for key, flag in optional_flags:
        value = cfg.get(key)
        if value is not None:
            cmd.extend([flag, str(value)])

    if bool(cfg.get("streaming", False)):
        cmd.append("--streaming")

    cmd.extend([str(x) for x in cfg.get("extra_args", [])])
    return cmd


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_json(config_path)
    cmd = build_cmd(cfg)
    working_dir = cfg.get("working_dir")
    cwd = str(Path(working_dir).expanduser().resolve()) if working_dir else None

    print("Running aiPerf command:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=cwd, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
