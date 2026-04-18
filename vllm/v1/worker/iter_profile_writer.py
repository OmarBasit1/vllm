# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write iteration profiling records from stdin to a JSONL output file."
        )
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Absolute or relative path to the output JSONL file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_file = os.path.abspath(os.path.expanduser(args.output_file))
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "a", encoding="utf-8", buffering=1) as fp:
        for line in sys.stdin:
            if not line:
                continue
            if line.endswith("\n"):
                fp.write(line)
            else:
                fp.write(line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
