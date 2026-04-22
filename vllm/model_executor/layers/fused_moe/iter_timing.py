# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _ExpertTimingPair:
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


_enabled = False
_iteration_active = False
_iteration_pairs: list[_ExpertTimingPair] = []
_last_pairs: tuple[_ExpertTimingPair, ...] = ()
_event_pool: list[_ExpertTimingPair] = []
_pool_idx = 0


def set_moe_iteration_timing_enabled(enabled: bool) -> None:
    global _enabled, _iteration_active, _last_pairs, _event_pool, _pool_idx
    _enabled = enabled
    _iteration_active = False
    _iteration_pairs.clear()
    _last_pairs = ()
    _event_pool.clear()
    _pool_idx = 0


def start_moe_iteration_timing() -> None:
    global _iteration_active, _pool_idx
    if not _enabled:
        return
    _iteration_active = True
    _iteration_pairs.clear()
    _pool_idx = 0


def finish_moe_iteration_timing_events(
) -> tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...]:
    global _iteration_active, _last_pairs
    if not _enabled:
        return ()

    # When Python wrappers are bypassed by cudagraph replay, no per-iteration
    # pairs may be captured. Reuse the most recently observed pair objects.
    if _iteration_pairs:
        _last_pairs = tuple(_iteration_pairs)
        pairs = tuple((p.start_event, p.end_event) for p in _iteration_pairs)
    else:
        pairs = tuple((p.start_event, p.end_event) for p in _last_pairs)

    _iteration_pairs.clear()
    _iteration_active = False
    return pairs


def record_moe_expert_start(_weight: torch.Tensor) -> _ExpertTimingPair | None:
    global _pool_idx
    if not _enabled or not _iteration_active:
        return None

    if _pool_idx >= len(_event_pool):
        _event_pool.append(
            _ExpertTimingPair(
                start_event=torch.cuda.Event(enable_timing=True),
                end_event=torch.cuda.Event(enable_timing=True),
            )
        )

    pair = _event_pool[_pool_idx]
    _pool_idx += 1
    _iteration_pairs.append(pair)

    pair.start_event.record()
    return pair


def record_moe_expert_end(handle: _ExpertTimingPair | None) -> None:
    if not _enabled or handle is None:
        return
    handle.end_event.record()
