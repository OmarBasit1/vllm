# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import queue
import threading
import zlib
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgspec
import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass
class _PendingMoERecord:
    iteration_id: int
    token_count: int
    layer_ids: list[int]
    request_token_counts: list[int]
    expert_ids_cpu: torch.Tensor
    done_event: torch.cuda.Event | None


class MoEProfiler:
    """Per-worker temporal MoE expert logger.

    Captures routed top-k expert ids at router time, groups them by
    request-position for each batch iteration, and writes periodic chunked
    compressed logs asynchronously to minimize impact on inference latency.
    """

    def __init__(
        self,
        log_dir: str,
        vllm_config: VllmConfig,
        flush_every: int = 64,
    ) -> None:
        self.vllm_config = vllm_config
        self._flush_every = max(1, flush_every)

        path = Path(log_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        self._log_dir = path

        self._device_expert_ids: torch.Tensor | None = None

        self._captured_layer_ids: set[int] = set()
        self._token_count: int = 0
        self._active_request_token_counts: list[int] = []

        self._active_iteration_id: int | None = None
        self._next_iteration_id: int = 0
        self._dp_slice: tuple[int, int] | None = None

        self._pending: deque[_PendingMoERecord] = deque()
        # Records can temporarily hold CPU tensors and are materialized by the
        # background writer thread into serializable nested lists.
        self._records: list[dict[str, Any]] = []
        self._chunk_index: int = 0
        self._session_timestamp_utc = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self._state_lock = threading.Lock()
        self._closed = False
        self._worker_error: Exception | None = None
        self.path: Path | None = None
        self._write_queue: queue.Queue[tuple[int, list[dict[str, Any]]] | None] = (
            queue.Queue(maxsize=0)
        )
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="moe-temporal-log-writer",
            daemon=True,
        )
        self._writer.start()

        pc = vllm_config.parallel_config
        self._metadata = {
            "format_version": 1,
            "log_type": "temporal_expert",
            "instance_id": vllm_config.instance_id,
            "model": vllm_config.model_config.model,
            "tensor_parallel_rank": pc.rank,
            "data_parallel_rank": pc.data_parallel_rank,
            "pipeline_parallel_size": pc.pipeline_parallel_size,
            "pid": os.getpid(),
            "session_timestamp_utc": self._session_timestamp_utc,
        }

    def init_buffer(
        self,
        max_num_batched_tokens: int,
    ) -> None:
        if self._device_expert_ids is not None:
            raise RuntimeError("MoE profiler buffer is already initialized.")

        hf_config = self.vllm_config.model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok

        self._device_expert_ids = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=torch.int32,
            device=current_platform.device_type,
        )

    def start_iteration(
        self,
        iteration_id: int | None = None,
        request_token_counts: list[int] | None = None,
    ) -> None:
        if iteration_id is None:
            iteration_id = self._next_iteration_id
            self._next_iteration_id += 1
        else:
            self._next_iteration_id = max(self._next_iteration_id, iteration_id + 1)

        self._active_iteration_id = iteration_id
        self._captured_layer_ids.clear()
        self._token_count = 0
        self._dp_slice = None
        if request_token_counts is None:
            self._active_request_token_counts = []
        else:
            self._active_request_token_counts = [
                int(x) for x in request_token_counts if int(x) > 0
            ]

    def log(
        self,
        layer_id: int,
        expert_ids: torch.Tensor,
        iteration_id: int | None = None,
        local_expert_map: torch.Tensor | None = None,
    ) -> None:
        if self._device_expert_ids is None:
            return

        if iteration_id is not None:
            self._active_iteration_id = iteration_id

        if self._active_iteration_id is None:
            return

        if layer_id >= self._device_expert_ids.shape[1] or layer_id < 0:
            return

        local_ids, token_count = self._slice_for_dp(expert_ids)
        if local_expert_map is not None:
            local_ids = self._filter_non_local_experts(local_ids, local_expert_map)

        if token_count <= 0:
            return

        token_count = min(token_count, self._device_expert_ids.shape[0])
        if local_ids.dtype != self._device_expert_ids.dtype:
            local_ids = local_ids.to(dtype=self._device_expert_ids.dtype)

        self._device_expert_ids[:token_count, layer_id, :].copy_(
            local_ids[:token_count, :],
            non_blocking=True,
        )

        self._captured_layer_ids.add(layer_id)
        self._token_count = max(self._token_count, token_count)

    def end_iteration(self) -> None:
        if self._active_iteration_id is None:
            return

        if not self._captured_layer_ids or self._token_count <= 0:
            self._active_iteration_id = None
            return

        if self._device_expert_ids is None:
            self._active_iteration_id = None
            return

        layer_ids = sorted(self._captured_layer_ids)
        layer_ids_t = torch.tensor(layer_ids, device=self._device_expert_ids.device)

        ids_gpu = torch.index_select(
            self._device_expert_ids[:self._token_count], dim=1, index=layer_ids_t
        ).contiguous()

        ids_cpu = torch.empty_like(
            ids_gpu,
            device="cpu",
            pin_memory=ids_gpu.is_cuda,
        )
        ids_cpu.copy_(ids_gpu, non_blocking=ids_gpu.is_cuda)

        local_request_token_counts = self._slice_request_token_counts_for_dp(
            self._active_request_token_counts
        )
        local_request_token_counts = self._normalize_request_token_counts(
            local_request_token_counts,
            self._token_count,
        )

        done_event: torch.cuda.Event | None = None
        if ids_gpu.is_cuda:
            done_event = torch.cuda.Event()
            done_event.record(torch.cuda.current_stream())

        self._pending.append(
            _PendingMoERecord(
                iteration_id=self._active_iteration_id,
                token_count=self._token_count,
                layer_ids=layer_ids,
                request_token_counts=local_request_token_counts,
                expert_ids_cpu=ids_cpu,
                done_event=done_event,
            )
        )

        self._active_iteration_id = None
        self._drain_pending(non_blocking=True)
        if len(self._records) >= self._flush_every:
            self.flush(force=False)

    def flush(self, force: bool = False) -> None:
        self._raise_if_worker_failed()
        self._drain_pending(non_blocking=not force)
        if not self._records:
            return

        chunk_index = self._chunk_index
        self._chunk_index += 1
        records = self._records
        self._records = []
        self._write_queue.put_nowait((chunk_index, records))

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return

        self.flush(force=True)

        with self._state_lock:
            self._closed = True

        self._write_queue.put(None)
        self._write_queue.join()
        self._writer.join(timeout=5.0)
        self._raise_if_worker_failed()

    def _slice_for_dp(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        ctx = get_forward_context()
        dp_meta = ctx.dp_metadata
        if dp_meta is None:
            return x, x.shape[0]

        if self._dp_slice is None:
            dp_rank = int(self.vllm_config.parallel_config.data_parallel_rank)
            token_num_per_dp = int(dp_meta.num_tokens_across_dp_cpu[dp_rank])
            cumsum = torch.cumsum(dp_meta.num_tokens_across_dp_cpu, dim=0)
            end_loc = int(cumsum[dp_rank])
            start_loc = end_loc - token_num_per_dp
            self._dp_slice = (start_loc, end_loc)
        else:
            start_loc, end_loc = self._dp_slice
            token_num_per_dp = end_loc - start_loc

        return x[start_loc:end_loc, :], token_num_per_dp

    @staticmethod
    def _filter_non_local_experts(
        expert_ids: torch.Tensor,
        local_expert_map: torch.Tensor,
    ) -> torch.Tensor:
        if expert_ids.numel() == 0:
            return expert_ids

        if local_expert_map.device != expert_ids.device:
            local_expert_map = local_expert_map.to(
                device=expert_ids.device,
                non_blocking=True,
            )

        expert_ids_i64 = expert_ids.to(dtype=torch.int64)
        valid_range = (expert_ids_i64 >= 0) & (
            expert_ids_i64 < local_expert_map.shape[0]
        )
        safe_ids = torch.where(
            valid_range,
            expert_ids_i64,
            torch.zeros_like(expert_ids_i64),
        )

        is_local = local_expert_map[safe_ids] >= 0
        keep_mask = valid_range & is_local
        filtered_ids = torch.where(
            keep_mask,
            expert_ids_i64,
            torch.full_like(expert_ids_i64, -1),
        )
        return filtered_ids

    def _slice_request_token_counts_for_dp(
        self, request_token_counts: list[int]
    ) -> list[int]:
        if not request_token_counts:
            return []

        if self._dp_slice is None:
            return [int(c) for c in request_token_counts if int(c) > 0]

        start_loc, end_loc = self._dp_slice
        cursor = 0
        local_counts: list[int] = []
        for count in request_token_counts:
            count = int(count)
            if count <= 0:
                continue
            next_cursor = cursor + count
            overlap = max(0, min(next_cursor, end_loc) - max(cursor, start_loc))
            if overlap > 0:
                local_counts.append(overlap)
            cursor = next_cursor
            if cursor >= end_loc:
                break

        return local_counts

    @staticmethod
    def _normalize_request_token_counts(
        request_token_counts: list[int], token_count: int
    ) -> list[int]:
        remaining = max(0, int(token_count))
        normalized: list[int] = []

        for count in request_token_counts:
            if remaining <= 0:
                break
            count = int(count)
            if count <= 0:
                continue
            used = min(count, remaining)
            normalized.append(used)
            remaining -= used

        if remaining > 0:
            normalized.append(remaining)

        return normalized

    def _drain_pending(self, non_blocking: bool) -> None:
        while self._pending:
            pending = self._pending[0]
            if pending.done_event is not None:
                if non_blocking and not pending.done_event.query():
                    break
                pending.done_event.synchronize()

            record = {
                "iteration_id": pending.iteration_id,
                "token_count": pending.token_count,
                "layer_ids": pending.layer_ids,
                "request_token_counts": pending.request_token_counts,
                "expert_ids": pending.expert_ids_cpu,
            }
            self._records.append(record)
            self._pending.popleft()

    def _raise_if_worker_failed(self) -> None:
        with self._state_lock:
            worker_error = self._worker_error
        if worker_error is not None:
            raise RuntimeError("Temporal MoE log writer thread failed") from worker_error

    def _writer_loop(self) -> None:
        encoder = msgspec.msgpack.Encoder()
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                chunk_index, records = item
                self._write_chunk(chunk_index, records, encoder)
            except Exception as exc:
                with self._state_lock:
                    if self._worker_error is None:
                        self._worker_error = exc
            finally:
                self._write_queue.task_done()

    def _get_chunk_paths(self, chunk_index: int) -> tuple[Path, Path]:
        file_name = (
            f"temporal_expert_profile_{self._metadata['instance_id']}_"
            f"dp{self._metadata['data_parallel_rank']}_"
            f"tp{self._metadata['tensor_parallel_rank']}_"
            f"pid{self._metadata['pid']}_"
            f"ts{self._session_timestamp_utc}_"
            f"chunk{chunk_index:06d}.msgpack.zlib"
        )
        target_path = self._log_dir / file_name
        temp_path = self._log_dir / f".{file_name}.tmp"
        return target_path, temp_path

    @classmethod
    def _materialize_iteration_record(cls, record: dict[str, Any]) -> dict[str, Any]:
        token_count = int(record["token_count"])
        layer_ids = [int(layer) for layer in record["layer_ids"]]
        request_token_counts = cls._normalize_request_token_counts(
            [int(c) for c in record.get("request_token_counts", [])],
            token_count,
        )

        expert_ids = record["expert_ids"]
        if not isinstance(expert_ids, torch.Tensor):
            expert_ids = torch.as_tensor(expert_ids)

        request_ranges: list[tuple[int, int]] = []
        start = 0
        for count in request_token_counts:
            end = start + count
            request_ranges.append((start, end))
            start = end

        layers: list[dict[str, Any]] = []
        for local_layer_idx, layer_no in enumerate(layer_ids):
            layer_experts = expert_ids[:, local_layer_idx, :]
            request_experts: list[list[list[int]]] = []
            for start, end in request_ranges:
                token_rows = layer_experts[start:end].tolist()
                request_experts.append(
                    [
                        [
                            int(expert_id)
                            for expert_id in token_row
                            if int(expert_id) >= 0
                        ]
                        for token_row in token_rows
                    ]
                )
            layers.append(
                {
                    "layer_no": int(layer_no),
                    "request_expert_ids": request_experts,
                }
            )

        return {
            "iteration_no": int(record["iteration_id"]),
            "token_count": token_count,
            "request_token_counts": request_token_counts,
            "layers": layers,
        }

    def _write_chunk(
        self,
        chunk_index: int,
        records: list[dict[str, Any]],
        encoder: msgspec.msgpack.Encoder,
    ) -> None:
        target_path, temp_path = self._get_chunk_paths(chunk_index)
        payload = {
            "metadata": self._metadata,
            "iterations": [
                self._materialize_iteration_record(record) for record in records
            ],
        }
        encoded = encoder.encode(payload)
        compressed = zlib.compress(encoded, level=5)
        with temp_path.open("wb") as f:
            f.write(compressed)
        os.replace(temp_path, target_path)
        with self._state_lock:
            self.path = target_path

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.debug("MoEProfiler cleanup failed", exc_info=True)
