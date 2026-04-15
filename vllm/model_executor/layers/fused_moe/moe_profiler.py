# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    expert_ids_cpu: torch.Tensor
    routing_weights_cpu: torch.Tensor | None
    done_event: torch.cuda.Event | None


class MoEProfiler:
    """Lightweight per-worker MoE routing profiler.

    Captures routed expert ids (and optional routing weights) at router time,
    buffers them on device, and asynchronously copies per-iteration snapshots
    to pinned CPU memory. Records are flushed in periodic binary chunks.
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
        self._device_routing_weights: torch.Tensor | None = None

        self._captured_layer_ids: set[int] = set()
        self._token_count: int = 0

        self._active_iteration_id: int | None = None
        self._next_iteration_id: int = 0
        self._dp_slice: tuple[int, int] | None = None

        self._pending: deque[_PendingMoERecord] = deque()
        self._records: list[dict[str, Any]] = []
        self._chunk_index: int = 0
        self._session_timestamp_utc = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        pc = vllm_config.parallel_config
        self._metadata = {
            "format_version": 1,
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

    def start_iteration(self, iteration_id: int | None = None) -> None:
        if iteration_id is None:
            iteration_id = self._next_iteration_id
            self._next_iteration_id += 1
        else:
            self._next_iteration_id = max(self._next_iteration_id, iteration_id + 1)

        self._active_iteration_id = iteration_id
        self._captured_layer_ids.clear()
        self._token_count = 0
        self._dp_slice = None

    def log(
        self,
        layer_id: int,
        expert_ids: torch.Tensor,
        iteration_id: int | None = None,
        routing_weights: torch.Tensor | None = None,
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
        if token_count <= 0:
            return

        token_count = min(token_count, self._device_expert_ids.shape[0])

        self._device_expert_ids[:token_count, layer_id, :].copy_(
            local_ids[:token_count, :],
            non_blocking=True,
        )

        if routing_weights is not None:
            local_weights, _ = self._slice_for_dp(routing_weights)
            if self._device_routing_weights is None:
                self._device_routing_weights = torch.zeros(
                    self._device_expert_ids.shape,
                    dtype=local_weights.dtype,
                    device=current_platform.device_type,
                )
            self._device_routing_weights[:token_count, layer_id, :].copy_(
                local_weights[:token_count, :],
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

        weights_cpu: torch.Tensor | None = None
        if self._device_routing_weights is not None:
            weights_gpu = torch.index_select(
                self._device_routing_weights[:self._token_count],
                dim=1,
                index=layer_ids_t,
            ).contiguous()
            weights_cpu = torch.empty_like(
                weights_gpu,
                device="cpu",
                pin_memory=weights_gpu.is_cuda,
            )
            weights_cpu.copy_(weights_gpu, non_blocking=weights_gpu.is_cuda)

        done_event: torch.cuda.Event | None = None
        if ids_gpu.is_cuda:
            done_event = torch.cuda.Event()
            done_event.record(torch.cuda.current_stream())

        self._pending.append(
            _PendingMoERecord(
                iteration_id=self._active_iteration_id,
                token_count=self._token_count,
                layer_ids=layer_ids,
                expert_ids_cpu=ids_cpu,
                routing_weights_cpu=weights_cpu,
                done_event=done_event,
            )
        )

        self._active_iteration_id = None
        self._drain_pending(non_blocking=True)
        if len(self._records) >= self._flush_every:
            self.flush(force=False)

    def flush(self, force: bool = False) -> None:
        self._drain_pending(non_blocking=not force)
        if not self._records:
            return

        file_name = (
            f"moe_profile_{self._metadata['instance_id']}_"
            f"dp{self._metadata['data_parallel_rank']}_"
            f"tp{self._metadata['tensor_parallel_rank']}_"
            f"pid{self._metadata['pid']}_"
            f"ts{self._session_timestamp_utc}_"
            f"chunk{self._chunk_index:06d}.pt"
        )
        out_path = self._log_dir / file_name

        payload = {
            "metadata": self._metadata,
            "iterations": self._records,
        }
        torch.save(payload, out_path)

        self._records = []
        self._chunk_index += 1

    def close(self) -> None:
        self.flush(force=True)

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
                "expert_ids": pending.expert_ids_cpu,
            }
            if pending.routing_weights_cpu is not None:
                record["routing_weights"] = pending.routing_weights_cpu
            self._records.append(record)
            self._pending.popleft()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            logger.debug("MoEProfiler cleanup failed", exc_info=True)
