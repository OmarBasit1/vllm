# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global cross-layer expert cache with single-pass (selectable 1/2 wave) compute.

One GPU pool of expert rows is shared across *all* MoE layers (only one layer
runs at a time). Experts a step needs that are already resident ("hits") cost no
transfer; the rest ("misses") are loaded on demand into the pool, evicting the
least-valuable resident experts. Because one layer's full working set
(``<= global_num_experts`` experts) always fits the pool, a layer never has to be
tiled into ``ceil(cold/width)`` waves — the wave count is a fixed knob:

* ``waves=1`` (stall): high-priority load *all* misses, sync, then one kernel over
  hits+misses.
* ``waves=2`` (overlap): wave 0 computes the resident hits while the copy stream
  fetches all misses; wave 1 computes the misses. The runner sums the two.

Residency is published by indirection: ``param.data`` points at the whole pool
once, ``local_num_experts = pool_experts`` is constant, and each wave registers an
``_expert_map`` resolving only that wave's experts to their pool slots (all others
map to ``-1`` and contribute exactly zero, so summing waves reconstructs the
full-residency result — same invariant the runner relies on).

A predicted set of upcoming-layer experts is prefetched on the same copy stream
*after* the current misses, throttled by a measured PCIe budget so speculation only
ever consumes transfer slack. Eviction is graded (LFU, with predicted-soon experts
held back to last resort) and can never evict an expert the current step needs: the
step's whole working set is pinned before any speculative eviction runs.

Eager-mode only: wave/eviction decisions inspect runtime ``topk_ids`` values, which
is incompatible with CUDA graph capture. Reuses ``_CpuParamOffloader`` from
``prefetch.py`` for the CPU-pinned storage half of the offload.
"""

import atexit
import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.prefetch import _CpuParamOffloader

logger = init_logger(__name__)

# (layer_name, global_expert_id) uniquely identifies a cached expert.
Key = tuple[str, int]


class ExpertPredictor(ABC):
    """Pluggable predictor for which experts an upcoming layer will need."""

    @abstractmethod
    def predict(self, layer_idx: int, k: int) -> list[int]:
        """Return up to k predicted global expert ids for the given layer."""
        raise NotImplementedError


class DummyFirstKPredictor(ExpertPredictor):
    """Static predictor: always predicts global expert ids 0..k-1.

    Ignores router logits, layer index, and request. The interface exists so a
    router-logit-aware predictor can be swapped in without touching the caching
    or streaming machinery; ``first_k`` is the placeholder shipped in v2.
    """

    def predict(self, layer_idx: int, k: int) -> list[int]:
        return list(range(k))


_PREDICTOR_REGISTRY: dict[str, type[ExpertPredictor]] = {
    "first_k": DummyFirstKPredictor,
}


def create_predictor(name: str) -> ExpertPredictor:
    if name not in _PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown expert-cache predictor {name!r}. "
            f"Available: {sorted(_PREDICTOR_REGISTRY)}"
        )
    return _PREDICTOR_REGISTRY[name]()


class EvictionPolicy(ABC):
    """Pluggable eviction ordering over resident keys."""

    @abstractmethod
    def rank(self, key: Key, freq: int, in_horizon: bool) -> tuple:
        """Sort key for a resident expert; *smaller ranks are evicted first*."""
        raise NotImplementedError


class LFUEviction(EvictionPolicy):
    """Graded least-frequently-used.

    Non-horizon experts are evicted before predicted-soon ("horizon") ones, and
    within each tier the least-frequently-used goes first. Protecting horizon
    experts as a last resort keeps a prefetched expert from being thrown out by
    an intervening layer's miss-load before its own layer runs — unless that
    layer's demand is large enough to exhaust every non-horizon victim, in which
    case a present need correctly outranks a future guess.
    """

    def rank(self, key: Key, freq: int, in_horizon: bool) -> tuple:
        return (in_horizon, freq)


_EVICTION_REGISTRY: dict[str, type[EvictionPolicy]] = {
    "lfu": LFUEviction,
}


def create_eviction(name: str) -> EvictionPolicy:
    if name not in _EVICTION_REGISTRY:
        raise ValueError(
            f"Unknown expert-cache eviction policy {name!r}. "
            f"Available: {sorted(_EVICTION_REGISTRY)}"
        )
    return _EVICTION_REGISTRY[name]()


def split_hits_misses(
    resident: set[int], needed: list[int]
) -> tuple[list[int], list[int]]:
    """Partition ``needed`` into experts already resident (hits) and not (misses).

    Every needed expert appears in exactly one of the two lists.
    """
    hits = [g for g in needed if g in resident]
    misses = [g for g in needed if g not in resident]
    return hits, misses


class _GlobalExpertCache:
    """One GPU pool of expert rows shared across all MoE layers.

    Single source of truth for residency: ``slot_to_key`` / ``key_to_slot`` and
    the per-param pool tensors are written only here. ``param.data`` is pointed at
    the whole pool once (in ``bind_params``) and never re-sliced; only the
    per-wave ``_expert_map`` changes.
    """

    def __init__(
        self,
        param_names: list[str],
        pool_experts: int,
        global_num_experts: int,
        device: torch.device,
        eviction: EvictionPolicy,
        per_expert_bytes: int,
    ):
        self.param_names = param_names
        self.pool_experts = pool_experts
        self.global_num_experts = global_num_experts
        self.device = device
        self.eviction = eviction
        self.per_expert_bytes = per_expert_bytes
        self._cuda = device.type == "cuda"

        # Per-layer CPU-pinned expert storage: layer_name -> {param -> (E, *shape)}.
        self.layer_cpu: dict[str, dict[str, torch.Tensor]] = {}
        # Per-layer routed_experts module (for publishing the expert map).
        self.layer_module: dict[str, nn.Module] = {}
        # The shared GPU pool, one tensor per param, allocated in build_pool.
        self.pool: dict[str, torch.Tensor] = {}
        # Contiguous per-wave compute buffer (dense rows the kernel sees).
        self.compute_buf: dict[str, torch.Tensor] = {}

        self.slot_to_key: list[Key | None] = [None] * pool_experts
        self.key_to_slot: dict[Key, int] = {}
        self.freq: dict[Key, int] = {}
        self.pinned_hot: set[Key] = set()  # always-on experts, never evicted
        self._step_pinned: set[Key] = set()  # current step's working set

        self._copy_stream: torch.cuda.Stream | None = None

    # -- setup -------------------------------------------------------------
    def register_layer(
        self,
        layer_name: str,
        module: nn.Module,
        cpu_storages: dict[str, torch.Tensor],
    ) -> None:
        self.layer_cpu[layer_name] = cpu_storages
        self.layer_module[layer_name] = module

    def build_pool(self) -> None:
        """Allocate the shared pool (scattered cache residency) and the contiguous
        per-wave compute buffer (dense rows the kernel actually reads).

        The kernel's ``expert_map`` path only supports *dense* local expert ids,
        so a wave gathers its experts from their scattered pool slots into
        ``compute_buf`` rows ``0..k-1`` and maps global id -> that dense row.
        ``compute_buf`` is sized to a full layer (only one wave runs at a time),
        reused across all layers and waves."""
        example = next(iter(self.layer_cpu.values()))
        for name in self.param_names:
            shape = tuple(example[name].shape[1:])
            self.pool[name] = torch.empty(
                (self.pool_experts,) + shape,
                dtype=example[name].dtype,
                device=self.device,
            )
            self.compute_buf[name] = torch.empty(
                (self.global_num_experts,) + shape,
                dtype=example[name].dtype,
                device=self.device,
            )
        if self._cuda:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        for module in self.layer_module.values():
            # Leave params pointing at a valid (empty) GPU view until first publish.
            for name in self.param_names:
                getattr(module, name).data = self.compute_buf[name][:0]
            module.local_num_experts = 0

    # -- residency ---------------------------------------------------------
    def is_resident(self, layer: str, gid: int) -> bool:
        return (layer, gid) in self.key_to_slot

    def bump(self, layer: str, gids: list[int]) -> None:
        for g in gids:
            key = (layer, g)
            self.freq[key] = self.freq.get(key, 0) + 1

    def pin_step(self, keys: set[Key]) -> None:
        self._step_pinned = keys

    def unpin_step(self) -> None:
        self._step_pinned = set()

    def _evictable_keys(self) -> list[Key]:
        protected = self.pinned_hot | self._step_pinned
        return [
            self.slot_to_key[s]  # type: ignore[misc]
            for s in range(self.pool_experts)
            if self.slot_to_key[s] is not None
            and self.slot_to_key[s] not in protected
        ]

    def _pick_victims(self, count: int, horizon: set[Key]) -> list[Key]:
        """Return up to ``count`` resident keys to evict, most-evictable first."""
        evictable = self._evictable_keys()
        return heapq.nsmallest(
            count,
            evictable,
            key=lambda k: self.eviction.rank(k, self.freq.get(k, 0), k in horizon),
        )

    def _free_slots(self) -> list[int]:
        return [s for s in range(self.pool_experts) if self.slot_to_key[s] is None]

    def load(self, keys: list[Key], horizon: set[Key]) -> "torch.cuda.Event | None":
        """Make ``keys`` resident, evicting as needed, and return a completion
        event (None on CPU / when nothing to load).

        Victims are drawn only from unpinned, non-current-step slots — so a load
        can never evict an expert the current step needs, nor an always-on hot
        expert. The fork-event discipline keeps a copy from overwriting a slot a
        prior wave's compute is still reading.
        """
        to_load = [k for k in keys if k not in self.key_to_slot]
        if not to_load:
            return None

        free = self._free_slots()
        need_evict = max(0, len(to_load) - len(free))
        victims = self._pick_victims(need_evict, horizon) if need_evict else []
        if len(free) + len(victims) < len(to_load):
            # Should not happen: pool_experts >= global_num_experts guarantees a
            # single layer fits, and prefetch is budgeted below that. Surface it.
            raise RuntimeError(
                f"ExpertCache pool exhausted: need {len(to_load)} slots but only "
                f"{len(free) + len(victims)} available (pinned "
                f"{len(self.pinned_hot) + len(self._step_pinned)})."
            )
        for vkey in victims:
            vslot = self.key_to_slot.pop(vkey)
            self.slot_to_key[vslot] = None
            free.append(vslot)

        if self._cuda:
            fork = torch.cuda.Event()
            torch.cuda.current_stream().record_event(fork)
            self._copy_stream.wait_event(fork)
            with torch.cuda.stream(self._copy_stream):
                self._copy_into(to_load, free)
            ev = torch.cuda.Event()
            ev.record(self._copy_stream)
            return ev
        self._copy_into(to_load, free)
        return None

    def _copy_into(self, keys: list[Key], free: list[int]) -> None:
        for (layer, gid), slot in zip(keys, free):
            for name in self.param_names:
                self.pool[name][slot].copy_(
                    self.layer_cpu[layer][name][gid], non_blocking=self._cuda
                )
            self.slot_to_key[slot] = (layer, gid)
            self.key_to_slot[(layer, gid)] = slot

    def publish(self, layer: str, gids: list[int]) -> None:
        """Gather ``gids`` from their scattered pool slots into contiguous
        ``compute_buf`` rows, point the module's params at that dense view, and
        register an ``_expert_map`` resolving global id -> its dense row (all
        other experts -> -1, contributing exactly zero)."""
        module = self.layer_module[layer]
        k = len(gids)
        expert_map = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int32, device=self.device
        )
        for j, g in enumerate(gids):
            slot = self.key_to_slot[(layer, g)]
            for name in self.param_names:
                self.compute_buf[name][j].copy_(
                    self.pool[name][slot], non_blocking=self._cuda
                )
            expert_map[g] = j
        for name in self.param_names:
            getattr(module, name).data = self.compute_buf[name][:k]
        module.local_num_experts = k
        module.register_buffer("_expert_map", expert_map)

    def resident_bytes(self) -> int:
        # Both the scattered pool and the contiguous compute buffer are GPU-resident.
        return (self.pool_experts + self.global_num_experts) * self.per_expert_bytes


class ExpertCacheOffloader(BaseOffloader):
    """Global cross-layer expert cache offloader.

    Args:
        pool_experts: Total experts the shared GPU pool holds (across all layers).
            Must be ``>= global_num_experts`` so any single layer fits.
        waves: 1 (stall: load all misses then one kernel) or 2 (compute hits while
            fetching misses, then compute misses).
        predict_k: Experts the predictor proposes per upcoming layer for prefetch.
        prefetch_horizon: How many layers ahead to prefetch.
        budget_gb: If > 0, assert measured GPU-resident pool bytes are within
            tolerance (memory-parity check). 0 disables.
        eviction_policy: Eviction policy identifier (``"lfu"``).
        predictor: Predictor identifier (``"first_k"``).
        offload_params: Parameter name segments to target (empty = all per-expert
            params of each MoE layer).
    """

    _BUDGET_TOLERANCE = 0.15
    _EMA_ALPHA = 0.3
    _BUDGET_SAFETY = 0.8

    def __init__(
        self,
        pool_experts: int,
        waves: int = 2,
        predict_k: int = 0,
        prefetch_horizon: int = 1,
        budget_gb: float = 0.0,
        eviction_policy: str = "lfu",
        predictor: str = "first_k",
        offload_params: set[str] | None = None,
    ):
        if waves not in (1, 2):
            raise ValueError(f"expert-cache waves must be 1 or 2, got {waves}")
        self.pool_experts = pool_experts
        self.waves = waves
        self.predict_k = predict_k
        self.prefetch_horizon = max(0, prefetch_horizon)
        self.budget_gb = budget_gb
        self.predictor = create_predictor(predictor)
        self.eviction = create_eviction(eviction_policy)
        self.offload_params = offload_params or set()

        self._param_offloaders: dict[str, dict[str, _CpuParamOffloader]] = {}
        self._routed_experts: dict[str, nn.Module] = {}
        self._devices: dict[str, torch.device] = {}
        self._layer_order: list[str] = []
        self._layer_index: dict[str, int] = {}
        self._cache: _GlobalExpertCache | None = None

        # Transfer scheduling state.
        self._t_per_expert_ms = 0.0  # from startup PCIe calibration
        self._layer_time_ema: dict[str, float | None] = {
            "prefill": None,
            "decode": None,
        }
        self._pending_timing: dict[str, tuple] = {}
        self._prev_prefetch_event: "torch.cuda.Event | None" = None

        self._resident_bytes = 0
        self.hot_hits = 0
        self.cold_streamed = 0
        self.wave_count = 0
        self.prefetch_loaded = 0
        self.bytes_streamed = 0

        # Per-bucket instrumentation: "prefill" = step contains at least one
        # prefill token (mixed prefill+decode under continuous batching);
        # "decode" = pure-decode step. hits/misses come from need (this
        # layer-step); prefetch_loaded comes from prediction (not needed yet).
        # `hits + misses` is every expert this step needed; `misses +
        # prefetch_loaded` is every actual fetch (H2D load) issued, split by
        # why it was issued.
        self.bucket_stats: dict[str, dict[str, int]] = {
            "prefill": {"hits": 0, "misses": 0, "prefetch_loaded": 0},
            "decode": {"hits": 0, "misses": 0, "prefetch_loaded": 0},
        }
        self._pass_count = 0
        self._STATS_LOG_INTERVAL = 50

    # -- module discovery (mirrors the prefetch/uva backends) --------------
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        all_modules = []
        for module in modules_generator:
            all_modules.append(module)
            for _, submodule in module.named_modules():
                if isinstance(submodule, MoERunner):
                    self._wrap_routed_experts(submodule.routed_experts)
        return all_modules

    def _wrap_routed_experts(self, routed_experts: nn.Module) -> None:
        layer_name = routed_experts.layer_name
        if layer_name in self._param_offloaders:
            return

        params = dict(routed_experts.named_parameters())
        if self.offload_params:
            target_names = [
                name
                for name in params
                if any(f".{p}." in f".{name}." for p in self.offload_params)
            ]
        else:
            target_names = list(params)
        if not target_names:
            return

        device = next(routed_experts.parameters()).device
        if device == torch.device("cpu"):
            return

        self._param_offloaders[layer_name] = {
            name: _CpuParamOffloader(module=routed_experts, param_name=name)
            for name in target_names
        }
        self._routed_experts[layer_name] = routed_experts
        self._devices[layer_name] = device
        self._layer_index[layer_name] = len(self._layer_order)
        self._layer_order.append(layer_name)

    def _validate_kernel_support(self, routed_experts: nn.Module) -> None:
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        moe_kernel = getattr(routed_experts.quant_method, "moe_kernel", None)
        fused_experts = None
        if moe_kernel is not None and not getattr(moe_kernel, "is_monolithic", True):
            fused_experts = moe_kernel.fused_experts
        if not isinstance(fused_experts, TritonExperts):
            kernel_name = (
                type(fused_experts).__name__ if fused_experts is not None else None
            )
            raise NotImplementedError(
                f"ExpertCacheOffloader only supports the Triton unquantized MoE "
                f"kernel (TritonExperts); layer {routed_experts.layer_name!r} "
                f"selected {kernel_name!r}. Other kernels either don't honor a "
                f"non-None expert_map or cache the local expert count at "
                f"construction time and would silently produce wrong output under "
                f"the pool-wide weight tensor."
            )

    def post_init(self) -> None:
        expert_shapes: dict[str, tuple] | None = None
        cache: _GlobalExpertCache | None = None
        per_expert_bytes = 0
        global_num_experts = 0

        for layer_name in self._layer_order:
            offloaders = self._param_offloaders[layer_name]
            routed_experts = self._routed_experts[layer_name]
            device = self._devices[layer_name]

            for offloader in offloaders.values():
                offloader.sync_cpu_storage()
            self._validate_kernel_support(routed_experts)

            cpu_storages = {
                name: offloader._cpu_storage for name, offloader in offloaders.items()
            }
            shapes = {name: tuple(s.shape[1:]) for name, s in cpu_storages.items()}
            if expert_shapes is None:
                expert_shapes = shapes
                global_num_experts = routed_experts.global_num_experts
                if self.pool_experts < global_num_experts:
                    raise ValueError(
                        f"--expert-cache-capacity ({self.pool_experts}) must be "
                        f">= the model's experts/layer ({global_num_experts}) so a "
                        f"single layer's working set always fits the pool."
                    )
                per_expert_bytes = sum(
                    s[0].numel() * s.element_size() for s in cpu_storages.values()
                )
                cache = _GlobalExpertCache(
                    param_names=list(offloaders.keys()),
                    pool_experts=self.pool_experts,
                    global_num_experts=global_num_experts,
                    device=device,
                    eviction=self.eviction,
                    per_expert_bytes=per_expert_bytes,
                )
            elif shapes != expert_shapes:
                raise NotImplementedError(
                    "ExpertCacheOffloader requires homogeneous expert shapes "
                    f"across MoE layers; layer {layer_name!r} has {shapes}, "
                    f"expected {expert_shapes}."
                )

            cache.register_layer(layer_name, routed_experts, cpu_storages)

        if cache is None:
            return
        cache.build_pool()
        self._cache = cache
        self._warm_always_on()
        self._resident_bytes = cache.resident_bytes()
        self._calibrate_pcie(per_expert_bytes)
        self._log_and_check_budget(global_num_experts, per_expert_bytes)
        atexit.register(self._log_final_stats)

    def _warm_always_on(self) -> None:
        """Pin architecturally always-activated routed experts (empty on a
        load-balanced router). Hook returns [] by default."""
        assert self._cache is not None
        for layer_name in self._layer_order:
            always_on = self._always_on_experts(layer_name)
            if not always_on:
                continue
            keys = [(layer_name, g) for g in always_on]
            self._cache.load(keys, horizon=set())
            self._cache.pinned_hot.update(keys)

    def _always_on_experts(self, layer_name: str) -> list[int]:
        return []

    def _calibrate_pcie(self, per_expert_bytes: int) -> None:
        assert self._cache is not None
        if not self._cache._cuda:
            return
        warm = min(16, self.pool_experts)
        example = next(iter(self._cache.layer_cpu.values()))
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        torch.cuda.synchronize()
        start.record()
        for name in self._cache.param_names:
            for s in range(warm):
                self._cache.pool[name][s].copy_(example[name][s], non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        self._t_per_expert_ms = start.elapsed_time(end) / max(1, warm)

    def _log_and_check_budget(
        self, global_num_experts: int, per_expert_bytes: int
    ) -> None:
        resident_gb = self._resident_bytes / 1e9
        full_gb = (
            len(self._layer_order) * global_num_experts * per_expert_bytes / 1e9
        )
        logger.info_once(
            "[ExpertCacheOffloader] Global pool: %d experts across %d MoE layers "
            "(waves=%d, predict_k=%d, horizon=%d, eviction=%s, predictor=%s, "
            "PCIe~%.3f ms/expert). Full expert weights in CPU pinned memory: "
            "%.4f GB. GPU-resident pool: %.4f GB.",
            self.pool_experts,
            len(self._layer_order),
            self.waves,
            self.predict_k,
            self.prefetch_horizon,
            type(self.eviction).__name__,
            type(self.predictor).__name__,
            self._t_per_expert_ms,
            full_gb,
            resident_gb,
        )
        if self.budget_gb <= 0:
            return
        deviation = abs(resident_gb - self.budget_gb) / self.budget_gb
        if deviation > self._BUDGET_TOLERANCE:
            raise ValueError(
                f"ExpertCacheOffloader GPU-resident pool {resident_gb:.3f} GB "
                f"deviates {deviation:.0%} from --expert-cache-budget-gb "
                f"{self.budget_gb:.3f} GB (tolerance {self._BUDGET_TOLERANCE:.0%}). "
                f"Adjust --expert-cache-capacity to hit parity."
            )

    # -- runtime -----------------------------------------------------------
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def _batch_bucket(self, topk_ids: torch.Tensor) -> str:
        """'prefill' if the batch contains any prefill tokens, else 'decode'.

        Best-effort from the forward context; falls back to num_tokens vs
        num_reqs, then to a single-token heuristic (safe in tests with no
        forward context)."""
        num_tokens = topk_ids.shape[0]
        try:
            from vllm.forward_context import get_forward_context

            md = get_forward_context().attn_metadata
            if isinstance(md, (list, tuple)):
                md = md[0] if md else None
            if isinstance(md, dict) and md:
                md = next(iter(md.values()))
            npt = getattr(md, "num_prefill_tokens", None)
            if npt is not None:
                return "prefill" if npt > 0 else "decode"
            nr = getattr(md, "num_reqs", None)
            if nr is not None:
                return "prefill" if num_tokens > nr else "decode"
        except Exception:
            pass
        return "decode" if num_tokens <= 1 else "prefill"

    def _horizon_keys(self, cur_idx: int) -> set[Key]:
        keys: set[Key] = set()
        for tl in range(
            cur_idx + 1, min(cur_idx + 1 + self.prefetch_horizon, len(self._layer_order))
        ):
            lname = self._layer_order[tl]
            for g in self.predictor.predict(tl, self.predict_k):
                keys.add((lname, g))
        return keys

    def _transfer_budget(self, bucket: str, num_tokens: int, num_misses: int) -> int:
        if self._t_per_expert_ms <= 0:
            return 0
        per_tok = self._layer_time_ema[bucket]
        if per_tok is None:
            return 0  # not warmed yet: no speculation until we can size it
        t_compute = per_tok * num_tokens
        t_miss = num_misses * self._t_per_expert_ms
        slack_ms = (t_compute - t_miss) * self._BUDGET_SAFETY
        if slack_ms <= 0:
            return 0
        return int(slack_ms / self._t_per_expert_ms)

    def _consume_timing(self, layer: str) -> None:
        pending = self._pending_timing.pop(layer, None)
        if pending is None:
            return
        bucket, num_tokens, start, end = pending
        if not end.query():
            return  # not finished; skip this sample rather than sync-stall
        elapsed = start.elapsed_time(end)
        per_tok = elapsed / max(1, num_tokens)
        prev = self._layer_time_ema[bucket]
        self._layer_time_ema[bucket] = (
            per_tok
            if prev is None
            else (1 - self._EMA_ALPHA) * prev + self._EMA_ALPHA * per_tok
        )

    def iter_expert_waves(
        self, routed_experts: nn.Module, topk_ids: torch.Tensor
    ) -> Generator[None, None, None]:
        cache = self._cache
        layer = routed_experts.layer_name
        if cache is None or layer not in self._layer_index:
            yield
            return

        cuda = cache._cuda
        if cuda and self._prev_prefetch_event is not None:
            torch.cuda.current_stream().wait_event(self._prev_prefetch_event)
            self._prev_prefetch_event = None
        self._consume_timing(layer)

        cur_idx = self._layer_index[layer]
        bucket = self._batch_bucket(topk_ids)
        num_tokens = topk_ids.shape[0]

        needed = torch.unique(topk_ids).tolist()
        resident = {g for g in needed if cache.is_resident(layer, g)}
        hits, misses = split_hits_misses(resident, needed)
        cache.bump(layer, needed)
        self.hot_hits += len(hits)
        self.cold_streamed += len(misses)
        self.bucket_stats[bucket]["hits"] += len(hits)
        self.bucket_stats[bucket]["misses"] += len(misses)

        # Reserve the whole current-iteration working set before any eviction.
        cache.pin_step({(layer, g) for g in needed})
        horizon = self._horizon_keys(cur_idx)
        miss_event = cache.load([(layer, g) for g in misses], horizon)
        self.bytes_streamed += len(misses) * cache.per_expert_bytes

        start_ev = end_ev = None
        if cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

        if self.waves == 1 or not hits:
            if cuda and miss_event is not None:
                torch.cuda.current_stream().wait_event(miss_event)
            cache.publish(layer, needed)
            if start_ev is not None:
                start_ev.record()
            self.wave_count += 1
            yield
        else:
            cache.publish(layer, hits)
            if start_ev is not None:
                start_ev.record()
            self.wave_count += 1
            yield  # wave 0 (hits) overlaps the miss copy
            if cuda and miss_event is not None:
                torch.cuda.current_stream().wait_event(miss_event)
            cache.publish(layer, misses)
            self.wave_count += 1
            yield

        if end_ev is not None:
            end_ev.record()
            self._pending_timing[layer] = (bucket, num_tokens, start_ev, end_ev)

        self._issue_prefetch(cur_idx, bucket, num_tokens, len(misses), horizon)
        cache.unpin_step()
        self._maybe_log_stats(cur_idx)

    def _issue_prefetch(
        self,
        cur_idx: int,
        bucket: str,
        num_tokens: int,
        num_misses: int,
        horizon: set[Key],
    ) -> None:
        assert self._cache is not None
        budget = self._transfer_budget(bucket, num_tokens, num_misses)
        if budget <= 0 or self.prefetch_horizon == 0:
            return
        # Candidate queue: predicted experts for upcoming layers only (entries for
        # the current or past layers are stale by construction and never enqueued).
        queue: deque[Key] = deque()
        for tl in range(
            cur_idx + 1, min(cur_idx + 1 + self.prefetch_horizon, len(self._layer_order))
        ):
            lname = self._layer_order[tl]
            for g in self.predictor.predict(tl, self.predict_k):
                queue.append((lname, g))
        batch: list[Key] = []
        while queue and len(batch) < budget:
            key = queue.popleft()
            if self._cache.is_resident(*key):  # promote-on-hit / already there
                continue
            batch.append(key)
        if not batch:
            return
        ev = self._cache.load(batch, horizon)
        self.prefetch_loaded += len(batch)
        self.bucket_stats[bucket]["prefetch_loaded"] += len(batch)
        self.bytes_streamed += len(batch) * self._cache.per_expert_bytes
        if ev is not None:
            self._prev_prefetch_event = ev

    def _bucket_metrics(self, bucket: str) -> dict[str, float]:
        s = self.bucket_stats[bucket]
        needed = s["hits"] + s["misses"]
        fetches = s["misses"] + s["prefetch_loaded"]
        return {
            "hits": s["hits"],
            "misses": s["misses"],
            "prefetch_loaded": s["prefetch_loaded"],
            "hit_rate": s["hits"] / needed if needed else 0.0,
            "prefetch_ratio": s["prefetch_loaded"] / fetches if fetches else 0.0,
        }

    def get_stats(self) -> dict:
        return {
            "hot_hits": self.hot_hits,
            "cold_streamed": self.cold_streamed,
            "prefetch_loaded": self.prefetch_loaded,
            "waves": self.wave_count,
            "bytes_streamed": self.bytes_streamed,
            "hit_rate": (
                self.hot_hits / (self.hot_hits + self.cold_streamed)
                if (self.hot_hits + self.cold_streamed)
                else 0.0
            ),
            "prefill": self._bucket_metrics("prefill"),
            "decode": self._bucket_metrics("decode"),
        }

    def _maybe_log_stats(self, cur_idx: int) -> None:
        """Emit a periodic stats line once per forward pass (layer 0 marks a new
        pass), so a long benchmark run's log has a running trail of bucketed
        hit-rate / prefetch-ratio without waiting for shutdown."""
        if cur_idx != 0:
            return
        self._pass_count += 1
        if self._pass_count % self._STATS_LOG_INTERVAL == 0:
            self._log_stats_line()

    def _log_stats_line(self) -> None:
        pf, dc = self._bucket_metrics("prefill"), self._bucket_metrics("decode")
        logger.info(
            "[ExpertCacheOffloader] stats (pass=%d) "
            "prefill+decode: hits=%d misses=%d prefetch=%d hit_rate=%.3f "
            "prefetch_ratio=%.3f | decode: hits=%d misses=%d prefetch=%d "
            "hit_rate=%.3f prefetch_ratio=%.3f",
            self._pass_count,
            pf["hits"], pf["misses"], pf["prefetch_loaded"],
            pf["hit_rate"], pf["prefetch_ratio"],
            dc["hits"], dc["misses"], dc["prefetch_loaded"],
            dc["hit_rate"], dc["prefetch_ratio"],
        )

    def _log_final_stats(self) -> None:
        # Registered via atexit; never let shutdown-time logging raise.
        try:
            logger.info("[ExpertCacheOffloader] final stats:")
            self._log_stats_line()
        except Exception:
            pass
