# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys


def test_iter_profile_writer_writes_jsonl(tmp_path):
    output_file = tmp_path / "iter_profile.jsonl"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.v1.worker.iter_profile_writer",
            "--output-file",
            str(output_file),
        ],
        stdin=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None

    expected_lines = [
        '{"iteration_idx":0,"batch_size":3,"latency_ms":1.2}',
        '{"iteration_idx":1,"batch_size":2,"latency_ms":0.8}',
    ]
    for line in expected_lines:
        proc.stdin.write(line + "\n")

    proc.stdin.close()
    assert proc.wait(timeout=5) == 0

    with output_file.open("r", encoding="utf-8") as f:
        written_lines = [line.strip() for line in f if line.strip()]

    assert written_lines == expected_lines
