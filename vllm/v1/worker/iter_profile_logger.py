# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_WRITE_QUEUE_SIZE = 8192
_WRITER_JOIN_TIMEOUT_S = 5.0
_WRITER_EXIT_TIMEOUT_S = 2.0


@dataclass
class IterProfileHandle:
    iteration_idx: int
    batch_size: int
    num_scheduled_tokens: int
    start_event: torch.cuda.Event
    start_time_ns: int


@dataclass
class _PendingIteration:
    iteration_idx: int
    batch_size: int
    num_scheduled_tokens: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    start_time_ns: int


class IterProfileLogger:
    """Low-overhead iteration latency logger backed by CUDA events.

    Records start/end CUDA events for each iteration, polls completed events
    without forcing stream/device synchronization, and forwards JSONL records
    to a dedicated writer subprocess.
    """

    def __init__(
        self,
        output_file: str,
        instance_id: str,
        global_rank: int,
        local_rank: int,
        dp_rank: int,
        tp_rank: int,
    ) -> None:
        self.output_file = output_file
        self._instance_id = instance_id
        self._global_rank = global_rank
        self._local_rank = local_rank
        self._dp_rank = dp_rank
        self._tp_rank = tp_rank

        self._next_iteration_idx = 0
        self._dropped_records = 0
        self._closed = False

        self._pending: deque[_PendingIteration] = deque()
        self._write_queue: queue.Queue[str | None] = queue.Queue(
            maxsize=_WRITE_QUEUE_SIZE
        )

        self._writer_proc = self._start_writer_process(output_file)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="IterProfileWriterDispatch",
        )
        self._writer_thread.start()

        logger.info(
            "Iteration profiler is enabled. Output file: %s",
            output_file,
        )

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        global_rank: int,
        local_rank: int,
    ) -> IterProfileLogger:
        output_dir = vllm_config.scheduler_config.iter_profile_dir
        if not output_dir:
            output_dir = os.path.join(tempfile.gettempdir(), "vllm_iter_profile")
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)

        parallel_config = vllm_config.parallel_config
        output_file = os.path.join(
            output_dir,
            (
                "iter_profile_"
                f"instance_{vllm_config.instance_id}_"
                f"rank_{global_rank}_"
                f"dp_{parallel_config.data_parallel_rank}_"
                f"tp_{parallel_config.rank}.jsonl"
            ),
        )

        return cls(
            output_file=output_file,
            instance_id=vllm_config.instance_id,
            global_rank=global_rank,
            local_rank=local_rank,
            dp_rank=parallel_config.data_parallel_rank,
            tp_rank=parallel_config.rank,
        )

    def start_iteration(
        self,
        batch_size: int,
        num_scheduled_tokens: int,
    ) -> IterProfileHandle:
        self.poll_ready_iterations()

        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        handle = IterProfileHandle(
            iteration_idx=self._next_iteration_idx,
            batch_size=batch_size,
            num_scheduled_tokens=num_scheduled_tokens,
            start_event=start_event,
            start_time_ns=time.time_ns(),
        )
        self._next_iteration_idx += 1
        return handle

    def finish_iteration(self, handle: IterProfileHandle) -> None:
        if self._closed:
            return

        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._pending.append(
            _PendingIteration(
                iteration_idx=handle.iteration_idx,
                batch_size=handle.batch_size,
                num_scheduled_tokens=handle.num_scheduled_tokens,
                start_event=handle.start_event,
                end_event=end_event,
                start_time_ns=handle.start_time_ns,
            )
        )

        self.poll_ready_iterations()

    def poll_ready_iterations(self) -> None:
        if self._closed:
            return

        while self._pending:
            pending = self._pending[0]
            if not pending.end_event.query():
                break

            latency_ms = float(pending.start_event.elapsed_time(pending.end_event))
            record = {
                "instance_id": self._instance_id,
                "global_rank": self._global_rank,
                "local_rank": self._local_rank,
                "dp_rank": self._dp_rank,
                "tp_rank": self._tp_rank,
                "iteration_idx": pending.iteration_idx,
                "batch_size": pending.batch_size,
                "num_scheduled_tokens": pending.num_scheduled_tokens,
                "latency_ms": latency_ms,
                "start_time_ns": pending.start_time_ns,
                "end_time_ns": time.time_ns(),
            }
            self._enqueue_record(record)
            self._pending.popleft()

    def close(self) -> None:
        if self._closed:
            return

        self.poll_ready_iterations()
        if self._pending:
            logger.warning(
                "Dropping %d pending iteration records at shutdown because "
                "their CUDA events are not ready.",
                len(self._pending),
            )
            self._pending.clear()

        self._closed = True
        self._signal_writer_shutdown()
        self._writer_thread.join(timeout=_WRITER_JOIN_TIMEOUT_S)

        if self._writer_thread.is_alive():
            logger.warning(
                "Iteration profiler writer thread did not exit in time; "
                "terminating writer process."
            )
            self._terminate_writer_process()

        if self._dropped_records > 0:
            logger.warning(
                "Dropped %d iteration profile records due to writer backpressure "
                "or writer errors.",
                self._dropped_records,
            )

    def _enqueue_record(self, record: dict[str, float | int | str]) -> None:
        line = json.dumps(record, separators=(",", ":"))
        try:
            self._write_queue.put_nowait(line)
        except queue.Full:
            self._dropped_records += 1

    def _start_writer_process(self, output_file: str) -> subprocess.Popen[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm.v1.worker.iter_profile_writer",
            "--output-file",
            output_file,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            close_fds=True,
        )
        if proc.stdin is None:
            raise RuntimeError("Failed to initialize iteration profile writer stdin.")
        return proc

    def _writer_loop(self) -> None:
        proc = self._writer_proc
        assert proc.stdin is not None

        while True:
            item = self._write_queue.get()
            if item is None:
                break

            try:
                proc.stdin.write(item)
                proc.stdin.write("\n")
            except (BrokenPipeError, OSError, ValueError):
                self._dropped_records += 1
                break

        with suppress(Exception):
            proc.stdin.close()

        if proc.poll() is None:
            try:
                proc.wait(timeout=_WRITER_EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._terminate_writer_process()

    def _signal_writer_shutdown(self) -> None:
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            with suppress(queue.Empty):
                self._write_queue.get_nowait()
            try:
                self._write_queue.put_nowait(None)
            except queue.Full:
                logger.warning("Failed to signal iteration profile writer shutdown.")

    def _terminate_writer_process(self) -> None:
        proc = self._writer_proc
        if proc.poll() is not None:
            return

        proc.terminate()
        try:
            proc.wait(timeout=_WRITER_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
