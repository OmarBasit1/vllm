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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.expert_predictor_mlp import ExpertPredictorMLP
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

    def observe(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> None:  # noqa: B027
        """Record this step's routing/hidden-state signal for ``layer_idx``.

        Called once per (layer, step) from ``iter_expert_waves``, before
        ``predict`` is queried for any upcoming layer. Default no-op; override
        in predictors that condition on runtime signal (e.g.
        ``HybridExpertPredictor``).
        """
        return


class DummyFirstKPredictor(ExpertPredictor):
    """Static predictor: always predicts global expert ids 0..k-1.

    Ignores router logits, layer index, and request. The interface exists so a
    router-logit-aware predictor can be swapped in without touching the caching
    or streaming machinery; ``first_k`` is the placeholder shipped in v2.
    """

    def predict(self, layer_idx: int, k: int) -> list[int]:
        return list(range(k))


class HybridExpertPredictor(ExpertPredictor):
    """Per-layer MLP (lookahead=1) with an unconditional-frequency fallback.

    For layer ``L``, ``predict(L, k)`` runs a per-layer ``ExpertPredictorMLP``
    on the layer-``(L-1)`` hidden state cached by the most recent
    ``observe(L-1, ...)`` call, aggregating the predicted per-token
    distributions across the step's tokens (summed probability mass) before
    taking the top-k -- a step's true "needed" set is the union of every
    token's routed experts, not any single token's. Runs unconditionally: no
    confidence gate, no fallback to a second predictor past this class.

    Falls back to that layer's marginal (unconditional expert-routing
    frequency, from the Markov table's cold-start column) whenever no
    layer-(L-1) hidden has been observed yet for this step -- always true for
    layer 0 (no predecessor exists), and also true for any layer beyond the
    1-hop lookahead the MLP was trained for if ``prefetch_horizon`` is
    configured deeper than 1.

    Loads all artifacts from ``artifacts_dir`` (see
    ``ExpertCacheOffloadConfig.predictor_artifacts``); MLP weights are moved
    to the compute device lazily, on the first ``observe`` call, since the
    device is not yet known at construction time (layer/device discovery in
    ``ExpertCacheOffloader`` happens after the predictor is created). The
    MLPs are owned only by this predictor -- never registered as
    ``routed_experts`` or listed in ``offload_params`` -- so neither the
    expert-cache pool nor ``--cpu-offload-gb`` can page them out; they stay
    GPU-resident for the life of the process.

    The per-layer forward is tiny but launch-bound (3 GEMMs + activations +
    topk = ~6 eager kernel launches, ~0.25 ms/call), which sits on the
    prefetch critical path. To cut that, the MLP+topk is captured as a
    replayable CUDA graph per (layer, token-count, dtype) -- collapsing those
    launches into one replay (~0.057 ms/call, measured 4.4x). Capture is lazy
    (first time a given shape is seen), bounded (``_MAX_GRAPHS`` total, only
    for token counts <= ``_MAX_GRAPH_BATCH`` so large prefill chunks stay on
    the eager path), and fail-safe: any capture error disables graphs globally
    and falls back to the plain eager forward, which is numerically identical.
    """

    _MAX_GRAPH_BATCH = 64
    _MAX_GRAPHS = 512

    def __init__(self, artifacts_dir: str) -> None:
        if not artifacts_dir:
            raise ValueError(
                "predictor='hybrid' requires --expert-cache-predictor-artifacts "
                "to point at a directory with markov_table.npz and "
                "mlp_ckpts/raw_hidden/layer_{L:02d}/best_model.pt checkpoints."
            )
        artifacts = Path(artifacts_dir)
        markov_path = artifacts / "markov_table.npz"
        if not markov_path.exists():
            raise FileNotFoundError(
                f"expert-cache hybrid predictor: {markov_path} not found"
            )
        markov = np.load(markov_path)
        self._marginal: np.ndarray = markov["marginal"]  # (num_layers, num_experts)

        self._mlps: dict[int, ExpertPredictorMLP] = {}
        ckpt_root = artifacts / "mlp_ckpts" / "raw_hidden"
        for layer_dir in sorted(ckpt_root.glob("layer_*")):
            ckpt_path = layer_dir / "best_model.pt"
            if not ckpt_path.exists():
                continue
            layer = int(layer_dir.name.split("_")[1])
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mlp = ExpertPredictorMLP(
                ckpt["input_dim"], ckpt["h1"], ckpt["h2"], ckpt["num_experts"]
            )
            mlp.load_state_dict(ckpt["state_dict"])
            mlp.eval()
            for p in mlp.parameters():
                p.requires_grad_(False)
            self._mlps[layer] = mlp
        if not self._mlps:
            raise FileNotFoundError(
                f"expert-cache hybrid predictor: no checkpoints found under {ckpt_root}"
            )

        self._device: torch.device | None = None
        self._hidden: dict[int, torch.Tensor] = {}

        # CUDA-graph cache: (layer_idx, batch, k, dtype) -> (graph, in, out).
        self._graphs: dict[tuple, tuple] = {}
        self._graph_pool = None
        self._cuda_graphs_ok = True

    def _ensure_device(self, device: torch.device) -> None:
        if self._device == device:
            return
        for mlp in self._mlps.values():
            mlp.to(device)
        self._device = device

    def observe(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> None:
        if hidden_states is None:
            return
        self._ensure_device(hidden_states.device)
        # Only the immediately-next layer's predict() needs this; a single
        # entry bounds memory regardless of prefetch_horizon.
        self._hidden = {layer_idx: hidden_states}

    def _marginal_topk(self, layer_idx: int, k: int) -> list[int]:
        layer_idx = min(layer_idx, self._marginal.shape[0] - 1)
        row = self._marginal[layer_idx]
        k = min(k, row.shape[0])
        return np.argsort(-row)[:k].tolist()

    @staticmethod
    def _mlp_topk(
        mlp: ExpertPredictorMLP, hidden: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Per-token expert-prob prediction aggregated (summed mass) across the
        step's tokens, returning the top-k expert ids as a GPU tensor."""
        probs = torch.softmax(mlp(hidden.to(torch.float32)), dim=-1)
        return torch.topk(probs.sum(dim=0), k).indices

    def _predict_eager(
        self, mlp: ExpertPredictorMLP, hidden: torch.Tensor, k: int
    ) -> list[int]:
        with torch.no_grad():
            return self._mlp_topk(mlp, hidden, k).tolist()

    def _capture_graph(
        self, mlp: ExpertPredictorMLP, hidden: torch.Tensor, k: int
    ) -> tuple | None:
        """Capture MLP+topk for this fixed (batch, dtype) as a replayable graph.
        Returns (graph, static_in, static_out) or None if capture fails (in
        which case graphs are disabled globally and callers use the eager
        path)."""
        try:
            static_in = torch.empty_like(hidden)
            static_in.copy_(hidden)
            warm = torch.cuda.Stream(device=hidden.device)
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm), torch.no_grad():
                for _ in range(3):
                    self._mlp_topk(mlp, static_in, k)
            torch.cuda.current_stream().wait_stream(warm)

            graph = torch.cuda.CUDAGraph()
            if self._graph_pool is None:
                capture = torch.cuda.graph(graph)
            else:
                capture = torch.cuda.graph(graph, pool=self._graph_pool)
            with capture, torch.no_grad():
                static_out = self._mlp_topk(mlp, static_in, k)
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            return (graph, static_in, static_out)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "expert-cache hybrid predictor: CUDA graph capture failed (%s); "
                "falling back to eager predict for all layers.",
                e,
            )
            self._cuda_graphs_ok = False
            return None

    def _predict_graphed(
        self, layer_idx: int, mlp: ExpertPredictorMLP, hidden: torch.Tensor, k: int
    ) -> list[int] | None:
        key = (layer_idx, hidden.shape[0], k, hidden.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= self._MAX_GRAPHS:
                return None
            entry = self._capture_graph(mlp, hidden, k)
            if entry is None:
                return None
            self._graphs[key] = entry
        graph, static_in, static_out = entry
        static_in.copy_(hidden)
        graph.replay()
        return static_out.tolist()

    def predict(self, layer_idx: int, k: int) -> list[int]:
        if layer_idx == 0:
            return self._marginal_topk(0, k)
        mlp = self._mlps.get(layer_idx)
        hidden = self._hidden.get(layer_idx - 1)
        if mlp is None or hidden is None:
            return self._marginal_topk(layer_idx, k)
        k = min(k, self._marginal.shape[1])
        if (
            self._cuda_graphs_ok
            and hidden.is_cuda
            and hidden.shape[0] <= self._MAX_GRAPH_BATCH
        ):
            out = self._predict_graphed(layer_idx, mlp, hidden, k)
            if out is not None:
                return out
        return self._predict_eager(mlp, hidden, k)


_PREDICTOR_REGISTRY: dict[str, type[ExpertPredictor]] = {
    "first_k": DummyFirstKPredictor,
    "hybrid": HybridExpertPredictor,
}


def create_predictor(name: str, artifacts_dir: str = "") -> ExpertPredictor:
    if name not in _PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown expert-cache predictor {name!r}. "
            f"Available: {sorted(_PREDICTOR_REGISTRY)}"
        )
    if name == "hybrid":
        return HybridExpertPredictor(artifacts_dir)
    return _PREDICTOR_REGISTRY[name]()


@dataclass(frozen=True)
class EvictionContext:
    """Key-independent state for one eviction decision (``_pick_victims``
    builds one and shares it across every candidate key's ``rank()`` call).

    Attributes:
        cur_layer_idx: Ordinal of the layer executing right now.
        num_layers: Total MoE layers, for the cyclic distance modulus.
        tick: Monotonic step counter (bumped once per ``bump()`` call) --
            "now", for recency.
        max_freq: Running max of ``_GlobalExpertCache.freq``, >= 1, for
            normalizing frequency into ``[0, 1]``.
    """

    cur_layer_idx: int
    num_layers: int
    tick: int
    max_freq: int


def _layer_staleness(layer_idx: int, ctx: EvictionContext) -> float:
    """Normalized cyclic forward distance from ``ctx.cur_layer_idx`` to
    ``layer_idx``, in ``[0, 1]``: ~0 = needed next (protect most), 1 = just
    executed (evict first). ``layer_idx == cur_layer_idx`` (distance 0, same
    layer as now) also lands near 0 -- a resident-but-not-currently-needed
    expert of the executing layer is exactly one full pass from its next
    possible use, the same as any other layer's "just wrapped around" case."""
    d = (layer_idx - ctx.cur_layer_idx) % ctx.num_layers
    return d / max(1, ctx.num_layers - 1)


class EvictionPolicy(ABC):
    """Pluggable eviction ordering over resident keys."""

    @abstractmethod
    def rank(
        self,
        key: Key,
        freq: int,
        last_used: int,
        in_horizon: bool,
        layer_idx: int,
        ctx: EvictionContext,
    ) -> tuple:
        """Sort key for a resident expert; *smaller ranks are evicted first*."""
        raise NotImplementedError


class LFUEviction(EvictionPolicy):
    """Graded least-frequently-used, optionally blended with layer-distance.

    Non-horizon experts are evicted before predicted-soon ("horizon") ones, and
    within each tier a blend of global frequency and layer-distance ranks
    evictability. ``distance_weight=0`` (default) is textbook graded-LFU: the
    least-frequently-used goes first, using the GLOBAL frequency (all layers,
    hits AND misses -- see ``_GlobalExpertCache.bump``), not a
    residency-scoped count. ``distance_weight>0`` additionally protects
    experts whose layer is about to execute and prioritizes evicting experts
    whose layer just executed (see ``_layer_staleness``). Protecting horizon
    experts as a last resort keeps a prefetched expert from being thrown out by
    an intervening layer's miss-load before its own layer runs — unless that
    layer's demand is large enough to exhaust every non-horizon victim, in which
    case a present need correctly outranks a future guess.
    """

    def __init__(self, distance_weight: float = 0.0):
        if not 0.0 <= distance_weight <= 1.0:
            raise ValueError(
                f"eviction distance_weight must be in [0, 1], got {distance_weight}"
            )
        self.distance_weight = distance_weight

    def rank(
        self,
        key: Key,
        freq: int,
        last_used: int,
        in_horizon: bool,
        layer_idx: int,
        ctx: EvictionContext,
    ) -> tuple:
        base = freq / ctx.max_freq
        w = self.distance_weight
        keep = (
            base
            if w == 0.0
            else (1 - w) * base + w * (1 - _layer_staleness(layer_idx, ctx))
        )
        return (in_horizon, keep)


class LRUEviction(EvictionPolicy):
    """Least-recently-used, optionally blended with layer-distance.

    Recency is measured in layer-steps (``_GlobalExpertCache._tick``, bumped
    once per ``bump()`` call), not wall-clock time. ``distance_weight``
    semantics mirror ``LFUEviction``.
    """

    def __init__(self, distance_weight: float = 0.0):
        if not 0.0 <= distance_weight <= 1.0:
            raise ValueError(
                f"eviction distance_weight must be in [0, 1], got {distance_weight}"
            )
        self.distance_weight = distance_weight

    def rank(
        self,
        key: Key,
        freq: int,
        last_used: int,
        in_horizon: bool,
        layer_idx: int,
        ctx: EvictionContext,
    ) -> tuple:
        base = last_used / ctx.tick
        w = self.distance_weight
        keep = (
            base
            if w == 0.0
            else (1 - w) * base + w * (1 - _layer_staleness(layer_idx, ctx))
        )
        return (in_horizon, keep)


class MRUEviction(EvictionPolicy):
    """Most-recently-used: the mirror image of LRU (evict the most recently
    used first), no distance term -- the deliberate control arm separating
    "recency direction" from "distance awareness" as independent axes."""

    def rank(
        self,
        key: Key,
        freq: int,
        last_used: int,
        in_horizon: bool,
        layer_idx: int,
        ctx: EvictionContext,
    ) -> tuple:
        return (in_horizon, 1.0 - last_used / ctx.tick)


_EVICTION_REGISTRY: dict[str, type[EvictionPolicy]] = {
    "lfu": LFUEviction,
    "lru": LRUEviction,
    "mru": MRUEviction,
}


def create_eviction(name: str, distance_weight: float = 0.0) -> EvictionPolicy:
    if name not in _EVICTION_REGISTRY:
        raise ValueError(
            f"Unknown expert-cache eviction policy {name!r}. "
            f"Available: {sorted(_EVICTION_REGISTRY)}"
        )
    if name == "mru":
        return MRUEviction()
    return _EVICTION_REGISTRY[name](distance_weight=distance_weight)


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

    _EMA_ALPHA = 0.3  # matches ExpertCacheOffloader._EMA_ALPHA (compute-time side)
    _DECAY_INTERVAL = 1000  # bump() calls between freq_decay applications
    # P7.5: dedicated (slower) decay for the fixed+marginal transfer-cost
    # regression -- ~20 effective samples of memory vs `_EMA_ALPHA`'s ~3.3,
    # since a 2-parameter fit needs more history to be statistically stable
    # than a plain running average does. See `_update_transfer_regression`.
    _XFER_REGRESSION_ALPHA = 0.05
    # Minimum effective sample weight and minimum coefficient of variation
    # (std/mean) of `n` required before trusting the fit -- rejects windows
    # that are merely non-degenerate (`denom` technically nonzero) but too
    # narrow a spread of batch sizes to pin down a slope reliably.
    _XFER_MIN_EFFECTIVE_SAMPLES = 5.0
    _XFER_MIN_CV = 0.15

    def __init__(
        self,
        param_names: list[str],
        pool_experts: int,
        global_num_experts: int,
        device: torch.device,
        eviction: EvictionPolicy,
        per_expert_bytes: int,
        freq_decay: float = 1.0,
        eviction_prefill: EvictionPolicy | None = None,
    ):
        self.param_names = param_names
        self.pool_experts = pool_experts
        self.global_num_experts = global_num_experts
        self.device = device
        self.eviction = eviction
        # P7.8: optional separate eviction policy for `bucket == "prefill"`
        # steps; `None` (default) reuses `eviction` for both buckets, so
        # existing configs/behavior are unchanged unless explicitly opted in.
        # See `_eviction_for`.
        self.eviction_prefill = eviction_prefill
        self.per_expert_bytes = per_expert_bytes
        self.freq_decay = freq_decay
        self._cuda = device.type == "cuda"

        # Per-layer CPU-pinned expert storage: layer_name -> {param -> (E, *shape)}.
        self.layer_cpu: dict[str, dict[str, torch.Tensor]] = {}
        # Per-layer routed_experts module (for publishing the expert map).
        self.layer_module: dict[str, nn.Module] = {}
        # The shared GPU pool, one tensor per param, allocated in build_pool.
        # `publish()` (P7.4 structural) points `param.data` at this tensor
        # directly and resolves `_expert_map` to pool slots -- no separate
        # per-wave compute buffer or gather.
        self.pool: dict[str, torch.Tensor] = {}
        # P7.12: one persistent `_expert_map` buffer per layer, allocated
        # once in `build_pool` and registered once; `publish()` refills it
        # in place (`fill_(-1)` + scatter) instead of allocating a fresh
        # tensor and re-registering a buffer every call. Safe because every
        # write and every kernel read of a given layer's map is issued on the
        # same (main) CUDA stream in program order -- the same invariant that
        # already lets `publish()` skip the P7.4 gather -- so a refill can
        # never race a still-in-flight read of the map it's replacing. Not
        # proven safe under a LoRA aux-stream schedule (`apply_w13_lora` also
        # reads `expert_map`, potentially on a separate stream joined back to
        # main later than this refill); this offloader isn't exercised with
        # LoRA today, but that combination should be re-verified before
        # relying on it in production.
        self._expert_map_buf: dict[str, torch.Tensor] = {}
        # layer_name -> ordinal, for the eviction distance term (`_layer_staleness`).
        self._layer_idx_of: dict[str, int] = {}

        self.slot_to_key: list[Key | None] = [None] * pool_experts
        self.key_to_slot: dict[Key, int] = {}
        # Incrementally maintained partition of slot indices (P7.7): avoids an
        # O(pool_experts) Python scan of `slot_to_key` on every `_free_slots()`/
        # `_pick_victims()` call. Updated at the two places occupancy actually
        # changes: `_copy_into` (occupy) and `load()`'s eviction loop (free).
        self._free_slots_set: set[int] = set(range(pool_experts))
        self._occupied_slots: set[int] = set()
        # Global (all-layer, hit+miss) usage stats feeding eviction ranking --
        # NOT residency-scoped, and never reset/deleted on eviction, so LFU/LRU
        # answer "globally hot/recent" even for currently-non-resident experts.
        self.freq: dict[Key, float] = {}
        self.last_used: dict[Key, int] = {}
        self._tick = 0  # monotonic, bumped once per `bump()` call; "now" for LRU/MRU
        self._max_freq = 0.0  # running max, for normalizing freq into [0, 1]
        self._step_pinned: set[Key] = set()  # current step's working set --
        # the ONLY eviction restriction; no separate "always-on" protection tier.

        self._copy_stream: torch.cuda.Stream | None = None
        # P7.12: lazily-created, reused across every `load()` call -- see
        # the reuse-safety comment at its use site in `load()`.
        self._fork_event: torch.cuda.Event | None = None

        # Real (live-measured) transfer-time tracking, replacing reliance on
        # the once-at-startup PCIe calibration (`_t_per_expert_ms`) -- see
        # `ExpertCacheOffloader._calibrate_pcie` and `_effective_t_per_expert_ms`.
        # A FIFO, drained opportunistically (never discarded -- CUDA events on
        # the same stream complete in program order, so checking only the
        # oldest entry is sufficient and correct).
        self._transfer_timing_pending: deque[
            tuple[int, torch.cuda.Event, torch.cuda.Event]
        ] = deque()
        # Lifetime sum/count (diagnostic only: an all-time average for
        # get_stats()/logging). `real_ms_per_expert_ema` is the value actually
        # fed back into `_transfer_budget` -- an EMA (not a lifetime average)
        # so it can track drift over a long-running server, matching the
        # design already used for the compute-time estimate.
        self.real_transfer_ms_sum = 0.0
        self.real_transfer_expert_count = 0
        self.real_ms_per_expert_ema: float | None = None

        # P7.5: exponentially-weighted least-squares fit of
        # `elapsed = fixed + marginal * n`, updated alongside
        # `real_ms_per_expert_ema` in `drain_transfer_timing`. Plain
        # `elapsed / n` (what `real_ms_per_expert_ema` computes) folds fixed
        # per-batch launch/sync overhead into what's treated as a purely
        # marginal per-expert cost -- measured systematically too high on the
        # small (~2-4 expert) batches this cache usually transfers, which
        # under-sizes `_transfer_budget` and keeps batches small: a
        # self-reinforcing pessimism loop. `_transfer_marginal_ms_ema` (the
        # fitted slope) is what `_effective_t_per_expert_ms` should use
        # instead once the fit has enough samples to be well-conditioned;
        # `_transfer_fixed_ms_ema` (the intercept) is diagnostic-only.
        #
        # P7.5 fix: the first shipped version decayed these sums at the same
        # rate as this class's other EMAs (`_EMA_ALPHA=0.3` -> ~3.3 effective
        # samples of memory), which is far too short a window for a 2-parameter
        # fit -- measured in production oscillating >2x between consecutive
        # log samples of the SAME run (e.g. 0.44/0.62/0.83/0.35 ms/expert on
        # one ShareGPT batch=2 run) while the plain average stayed flat, and
        # the only existing guard (`abs(denom) < 1e-6`) only rejects an
        # EXACTLY-degenerate window, not a merely noisy/ill-conditioned one.
        # Fixed two ways: (1) a slower, dedicated decay
        # (`_XFER_REGRESSION_ALPHA`, ~20 effective samples) so the fit
        # averages over enough transfers to be stable, and (2) an explicit
        # relative-variance gate on `n` (see `_update_transfer_regression`)
        # so a window where recent batch sizes happen to be nearly identical
        # -- technically non-degenerate but still a poor basis for a 2-param
        # fit -- is rejected rather than trusted.
        self._xfer_ew_n = 0.0
        self._xfer_ew_e = 0.0
        self._xfer_ew_n2 = 0.0
        self._xfer_ew_ne = 0.0
        self._xfer_ew_w = 0.0
        self._transfer_marginal_ms_ema: float | None = None
        self._transfer_fixed_ms_ema: float | None = None

    # -- setup -------------------------------------------------------------
    def register_layer(
        self,
        layer_name: str,
        module: nn.Module,
        cpu_storages: dict[str, torch.Tensor],
        layer_idx: int,
    ) -> None:
        self.layer_cpu[layer_name] = cpu_storages
        self.layer_module[layer_name] = module
        self._layer_idx_of[layer_name] = layer_idx

    def build_pool(self) -> None:
        """Allocate the shared pool (scattered cache residency), one tensor
        per param. `publish()` (P7.4 structural) points every layer's
        `param.data` at this same pool tensor directly -- no per-wave gather
        or separate compute buffer; only the wave's `_expert_map` (global id
        -> pool slot, or -1) changes. Each layer's `_expert_map` buffer (P7.12)
        is also allocated and registered once here; `publish()` refills it in
        place rather than allocating a fresh tensor per call."""
        example = next(iter(self.layer_cpu.values()))
        for name in self.param_names:
            shape = tuple(example[name].shape[1:])
            self.pool[name] = torch.empty(
                (self.pool_experts,) + shape,
                dtype=example[name].dtype,
                device=self.device,
            )
        if self._cuda:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        for layer_name, module in self.layer_module.items():
            # Leave params pointing at a valid (empty) GPU view and no
            # resident experts until the first `publish()` call for that
            # layer.
            for name in self.param_names:
                getattr(module, name).data = self.pool[name][:0]
            module.local_num_experts = 0
            expert_map = torch.full(
                (self.global_num_experts,), -1, dtype=torch.int32, device=self.device
            )
            self._expert_map_buf[layer_name] = expert_map
            module.register_buffer("_expert_map", expert_map)

    # -- residency ---------------------------------------------------------
    def is_resident(self, layer: str, gid: int) -> bool:
        return (layer, gid) in self.key_to_slot

    def bump(self, layer: str, gids: list[int]) -> None:
        self._tick += 1
        if self.freq_decay != 1.0 and self._tick % self._DECAY_INTERVAL == 0:
            for k in self.freq:
                self.freq[k] *= self.freq_decay
            self._max_freq *= self.freq_decay
        for g in gids:
            key = (layer, g)
            freq = self.freq.get(key, 0) + 1
            self.freq[key] = freq
            self.last_used[key] = self._tick
            if freq > self._max_freq:
                self._max_freq = freq

    def pin_step(self, keys: set[Key]) -> None:
        self._step_pinned = keys

    def unpin_step(self) -> None:
        self._step_pinned = set()

    def _evictable_keys(self) -> list[Key]:
        return [
            self.slot_to_key[s]  # type: ignore[misc]
            for s in self._occupied_slots
            if self.slot_to_key[s] not in self._step_pinned
        ]

    def _eviction_for(self, bucket: str) -> EvictionPolicy:
        """Resolve the `EvictionPolicy` for `bucket` (P7.8): `eviction_prefill`
        when set and `bucket == "prefill"`, else the default `eviction`."""
        if bucket == "prefill" and self.eviction_prefill is not None:
            return self.eviction_prefill
        return self.eviction

    def _pick_victims(
        self, count: int, horizon: set[Key], cur_layer_idx: int, bucket: str = "decode"
    ) -> list[Key]:
        """Return up to ``count`` resident keys to evict, most-evictable first."""
        if count <= 0:
            return []
        ctx = EvictionContext(
            cur_layer_idx=cur_layer_idx,
            num_layers=max(1, len(self._layer_idx_of)),
            tick=max(1, self._tick),
            max_freq=max(1, self._max_freq),
        )
        ev = self._eviction_for(bucket)
        fast = self._pick_victims_fast(count, horizon, ctx, ev)
        if fast is not None:
            return fast
        return self._pick_victims_slow(count, horizon, ctx, ev)

    def _pick_victims_slow(
        self,
        count: int,
        horizon: set[Key],
        ctx: "EvictionContext",
        ev: EvictionPolicy,
    ) -> list[Key]:
        """Reference path: per-candidate `EvictionPolicy.rank()` via
        `heapq.nsmallest`. Kept as the ground truth for any `EvictionPolicy`
        the vectorized fast path below doesn't recognize (only the three
        registered types are vectorized)."""
        evictable = self._evictable_keys()
        return heapq.nsmallest(
            count,
            evictable,
            key=lambda k: ev.rank(
                k,
                self.freq.get(k, 0),
                self.last_used.get(k, 0),
                k in horizon,
                self._layer_idx_of[k[0]],
                ctx,
            ),
        )

    def _pick_victims_fast(
        self,
        count: int,
        horizon: set[Key],
        ctx: "EvictionContext",
        ev: EvictionPolicy,
    ) -> list[Key] | None:
        """Vectorized victim selection (P7.7) for the three shipped policies,
        replacing a per-candidate Python `rank()` virtual-dispatch (plus
        `heapq.nsmallest` bookkeeping) with a single gather into numpy arrays
        followed by vectorized arithmetic and `np.argpartition`.

        `freq`/`last_used` are read fresh from the canonical dicts every call
        rather than cached per-slot: either dict can be mutated directly (not
        only via `bump()` -- e.g. tests do this deliberately), so caching
        would risk silently ranking on stale values. The gather loop itself
        (O(occupied slots)) is unavoidable for that reason, but it replaces
        the far more expensive part: a virtual `rank()` call plus tuple
        construction per candidate. Returns `None` (caller falls back to
        `_pick_victims_slow`) for any `EvictionPolicy` not one of the three
        registered types.
        """
        is_lfu = isinstance(ev, LFUEviction)
        is_lru = isinstance(ev, LRUEviction)
        is_mru = isinstance(ev, MRUEviction)
        if not (is_lfu or is_lru or is_mru):
            return None

        keys = [
            self.slot_to_key[s]  # type: ignore[misc]
            for s in self._occupied_slots
            if self.slot_to_key[s] not in self._step_pinned
        ]
        n = len(keys)
        if n == 0:
            return []

        last_used_arr = np.fromiter(
            (self.last_used.get(k, 0) for k in keys), dtype=np.float64, count=n
        )
        in_horizon_arr = np.fromiter(
            (k in horizon for k in keys), dtype=bool, count=n
        )

        if is_mru:
            keep = 1.0 - last_used_arr / ctx.tick
        else:
            if is_lfu:
                freq_arr = np.fromiter(
                    (self.freq.get(k, 0) for k in keys), dtype=np.float64, count=n
                )
                base = freq_arr / ctx.max_freq
            else:
                base = last_used_arr / ctx.tick
            w = ev.distance_weight
            if w == 0.0:
                keep = base
            else:
                layer_idx_arr = np.fromiter(
                    (self._layer_idx_of[k[0]] for k in keys), dtype=np.int64, count=n
                )
                staleness = (
                    (layer_idx_arr - ctx.cur_layer_idx) % ctx.num_layers
                ) / max(1, ctx.num_layers - 1)
                keep = (1 - w) * base + w * (1 - staleness)

        non_horizon_idx = np.flatnonzero(~in_horizon_arr)
        horizon_idx = np.flatnonzero(in_horizon_arr)
        picks = self._smallest_indices(non_horizon_idx, keep, count)
        if len(picks) < count:
            picks += self._smallest_indices(horizon_idx, keep, count - len(picks))
        return [keys[i] for i in picks]

    @staticmethod
    def _smallest_indices(idx: np.ndarray, keep: np.ndarray, count: int) -> list[int]:
        """Indices (into `keep`) of the `count` smallest values among `idx`,
        ascending. Uses `argpartition` (O(n)) instead of a full sort when a
        strict subset is requested."""
        if idx.size == 0 or count <= 0:
            return []
        if count >= idx.size:
            ordered = idx[np.argsort(keep[idx], kind="stable")]
        else:
            top = idx[np.argpartition(keep[idx], count - 1)[:count]]
            ordered = top[np.argsort(keep[top], kind="stable")]
        return ordered.tolist()

    def _free_slots(self) -> list[int]:
        return list(self._free_slots_set)

    def load(
        self,
        keys: list[Key],
        horizon: set[Key],
        cur_layer_idx: int = 0,
        bucket: str = "decode",
    ) -> "torch.cuda.Event | None":
        """Make ``keys`` resident, evicting as needed, and return a completion
        event (None on CPU / when nothing to load).

        Victims are drawn only from non-current-step slots (``_step_pinned``)
        — so a load can never evict an expert the current step needs; every
        other resident expert is a fair eviction candidate. The fork-event
        discipline keeps a copy from overwriting a slot a prior wave's compute
        is still reading. ``bucket`` selects the eviction policy (P7.8; see
        ``_eviction_for``).
        """
        to_load = [k for k in keys if k not in self.key_to_slot]
        if not to_load:
            return None

        free = self._free_slots()
        need_evict = max(0, len(to_load) - len(free))
        victims = (
            self._pick_victims(need_evict, horizon, cur_layer_idx, bucket)
            if need_evict
            else []
        )
        if len(free) + len(victims) < len(to_load):
            # Should not happen: pool_experts >= global_num_experts guarantees a
            # single layer fits, and prefetch is budgeted below that. Surface it.
            raise RuntimeError(
                f"ExpertCache pool exhausted: need {len(to_load)} slots but only "
                f"{len(free) + len(victims)} available (pinned "
                f"{len(self._step_pinned)})."
            )
        for vkey in victims:
            vslot = self.key_to_slot.pop(vkey)
            self.slot_to_key[vslot] = None
            self._occupied_slots.discard(vslot)
            self._free_slots_set.add(vslot)
            free.append(vslot)

        if self._cuda:
            # P7.12: reuse one persistent non-timing Event instead of
            # allocating a fresh one per `load()` call -- safe because it is
            # recorded and immediately consumed (via `wait_event`, which only
            # needs the event's *next* recorded point, not its prior one)
            # within this same call; nothing holds a reference to it across
            # calls the way `_transfer_timing_pending` does for the timing
            # events below.
            if self._fork_event is None:
                self._fork_event = torch.cuda.Event()
            fork = self._fork_event
            torch.cuda.current_stream().record_event(fork)
            self._copy_stream.wait_event(fork)
            xfer_start = torch.cuda.Event(enable_timing=True)
            xfer_start.record(self._copy_stream)
            with torch.cuda.stream(self._copy_stream):
                self._copy_into(to_load, free)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(self._copy_stream)
            self._transfer_timing_pending.append((len(to_load), xfer_start, ev))
            return ev
        self._copy_into(to_load, free)
        return None

    def drain_transfer_timing(self) -> None:
        """Accumulate real (live-measured) H2D transfer time for completed
        loads. Updates both the lifetime `real_transfer_ms_sum`/
        `real_transfer_expert_count` (diagnostic-only, an all-time average)
        and `real_ms_per_expert_ema` (an EMA, like the compute-time estimate,
        so it tracks drift over a long-running server -- this is the value
        `_transfer_budget` actually uses). Events on the copy stream complete
        in program order, so checking only the oldest pending entry is
        sufficient -- if it isn't ready, none behind it are either. Never
        discards a sample; unready entries stay queued."""
        while self._transfer_timing_pending:
            n, start, end = self._transfer_timing_pending[0]
            if not end.query():
                break
            self._transfer_timing_pending.popleft()
            elapsed = start.elapsed_time(end)
            self.real_transfer_ms_sum += elapsed
            self.real_transfer_expert_count += n
            ms_per_expert = elapsed / max(1, n)
            prev = self.real_ms_per_expert_ema
            self.real_ms_per_expert_ema = (
                ms_per_expert
                if prev is None
                else (1 - self._EMA_ALPHA) * prev + self._EMA_ALPHA * ms_per_expert
            )
            self._update_transfer_regression(n, elapsed)

    def _update_transfer_regression(self, n: int, elapsed_ms: float) -> None:
        """Exponentially-weighted least-squares fit of
        `elapsed = fixed + marginal * n` (P7.5). See the field docstring in
        `__init__` for why this is needed, and for why the decay/gating below
        (P7.5 fix) is stricter than the first shipped version."""
        decay = 1 - self._XFER_REGRESSION_ALPHA
        self._xfer_ew_n = decay * self._xfer_ew_n + n
        self._xfer_ew_e = decay * self._xfer_ew_e + elapsed_ms
        self._xfer_ew_n2 = decay * self._xfer_ew_n2 + n * n
        self._xfer_ew_ne = decay * self._xfer_ew_ne + n * elapsed_ms
        self._xfer_ew_w = decay * self._xfer_ew_w + 1.0

        if self._xfer_ew_w < self._XFER_MIN_EFFECTIVE_SAMPLES:
            return
        mean_n = self._xfer_ew_n / self._xfer_ew_w
        if mean_n <= 0:
            return
        var_n = max(0.0, self._xfer_ew_n2 / self._xfer_ew_w - mean_n * mean_n)
        # Coefficient of variation of `n`: rejects windows where recent batch
        # sizes are too similar to pin down a 2-parameter fit reliably, even
        # though the raw `denom` below is technically nonzero.
        if var_n**0.5 / mean_n < self._XFER_MIN_CV:
            return
        denom = self._xfer_ew_w * self._xfer_ew_n2 - self._xfer_ew_n * self._xfer_ew_n
        if abs(denom) < 1e-6:
            return
        slope = (
            self._xfer_ew_w * self._xfer_ew_ne - self._xfer_ew_n * self._xfer_ew_e
        ) / denom
        intercept = (self._xfer_ew_e - slope * self._xfer_ew_n) / self._xfer_ew_w
        # Marginal cost per expert cannot be negative; a noisy/unstable fit
        # falls back to `real_ms_per_expert_ema` rather than propagate a
        # nonsensical (or negative) budget input.
        if slope > 0:
            self._transfer_marginal_ms_ema = slope
            self._transfer_fixed_ms_ema = max(0.0, intercept)

    def _copy_into(self, keys: list[Key], free: list[int]) -> None:
        for (layer, gid), slot in zip(keys, free):
            for name in self.param_names:
                self.pool[name][slot].copy_(
                    self.layer_cpu[layer][name][gid], non_blocking=self._cuda
                )
            self.slot_to_key[slot] = (layer, gid)
            self.key_to_slot[(layer, gid)] = slot
            self._free_slots_set.discard(slot)
            self._occupied_slots.add(slot)

    def publish(self, layer: str, gids: list[int]) -> None:
        """Point the module's params at the shared pool directly and register
        an ``_expert_map`` resolving each of ``gids`` to its pool slot (every
        other global id -> -1, contributing exactly zero).

        P7.4 (structural): no per-wave gather. Earlier revisions copied
        ``gids`` out of their scattered pool slots into a dense per-layer
        ``compute_buf`` and mapped global id -> that dense row; the kernel's
        ``expert_map`` indirection needs no such densification; it already
        resolves an arbitrary "local" index per global id (used elsewhere for
        expert-parallel sharding), so mapping straight to the pool slot and
        pointing ``param.data`` at the whole pool (constant across waves) is
        equally valid and skips the gather kernels entirely --
        `_prepare_expert_assignment` pads/sizes its per-expert blocks off
        ``global_num_experts``, not ``local_num_experts``, so a larger
        (pool-sized) local weight tensor
        does not add per-token compute. Safe under the existing residency
        invariants: eviction can never touch a slot this wave's ``gids``
        occupy (the whole step's working set is pinned before any load), and
        the fork-event/`_wait_for_pending_prefetch` discipline already
        guarantees a slot is never overwritten while a still-in-flight kernel
        from an earlier wave/layer might read it -- both apply identically
        whether the kernel reads a gathered copy or the pool slot itself.

        P7.12: the ``_expert_map`` buffer is refilled in place (one persistent
        tensor per layer, allocated in ``build_pool``) rather than allocated
        fresh and re-registered every call -- safe for the same single-stream,
        program-order reason the pool-slot indirection above is (see the
        `_expert_map_buf` field docstring for the one known caveat, LoRA's
        aux-stream schedule).
        """
        module = self.layer_module[layer]
        expert_map = self._expert_map_buf[layer]
        expert_map.fill_(-1)
        if gids:
            slots = torch.tensor(
                [self.key_to_slot[(layer, g)] for g in gids],
                dtype=torch.int32,
                device=self.device,
            )
            gid_tensor = torch.tensor(gids, dtype=torch.long, device=self.device)
            expert_map[gid_tensor] = slots
        for name in self.param_names:
            getattr(module, name).data = self.pool[name]
        module.local_num_experts = self.pool_experts

    def resident_bytes(self) -> int:
        # P7.4 (structural): only the scattered pool is GPU-resident now --
        # no separate compute buffer.
        return self.pool_experts * self.per_expert_bytes


class ExpertCacheOffloader(BaseOffloader):
    """Global cross-layer expert cache offloader.

    Args:
        pool_experts: Total experts the shared GPU pool holds (across all layers).
            Must be ``>= global_num_experts`` so any single layer fits.
        waves: 1 (stall: load all misses then one kernel) or 2 (compute hits while
            fetching misses, then compute misses).
        predict_k: Experts the predictor proposes per upcoming layer for prefetch.
            ``-1`` (default): auto-size to the observed per-bucket working-set
            EMA once warmed (else ``num_tokens * top_k``), capped at
            ``global_num_experts``. ``0`` disables prediction. A
            positive int is a static override.
        prefetch_horizon: How many layers ahead to prefetch.
        budget_gb: If > 0, assert measured GPU-resident pool bytes are within
            tolerance (memory-parity check). 0 disables.
        eviction_policy: Eviction policy identifier (``"lfu"``, ``"lru"``, or
            ``"mru"``).
        predictor: Predictor identifier (``"first_k"``).
        budget_safety: Fraction of the estimated copy-free window trusted for
            speculative prefetch (see ``_transfer_budget``).
        eviction_distance_weight: Blend weight (``[0, 1]``) for layer-distance
            awareness in ``lfu``/``lru`` (ignored by ``mru``); 0 = textbook
            policy, see ``LFUEviction``/``LRUEviction``.
        eviction_freq_decay: Geometric decay applied to all ``freq`` counts
            every ``_GlobalExpertCache._DECAY_INTERVAL`` steps; 1.0 (default)
            disables decay and reproduces the original never-decaying counts.
        eviction_policy_prefill: (P7.8) Eviction policy identifier used only
            for ``bucket == "prefill"`` steps; ``None`` (default) reuses
            ``eviction_policy`` for both buckets. Prefill's near-full-range
            horizon protection (P7.3) puts eviction pressure almost entirely
            on the horizon tier, where recency alone can correlate poorly
            with per-token need order -- a distinct policy is a candidate for
            that regime specifically.
        eviction_distance_weight_prefill: Distance-weight override paired
            with ``eviction_policy_prefill``; ``None`` (default) reuses
            ``eviction_distance_weight``. Ignored if
            ``eviction_policy_prefill`` is unset or ``"mru"``.
        offload_params: Parameter name segments to target (empty = all per-expert
            params of each MoE layer).
    """

    _BUDGET_TOLERANCE = 0.15
    _EMA_ALPHA = 0.3
    _MAX_EVENT_POOL = 256  # P7.12 pooled-Event free-list cap

    def __init__(
        self,
        pool_experts: int,
        waves: int = 2,
        predict_k: int = -1,
        prefetch_horizon: int = 1,
        budget_gb: float = 0.0,
        eviction_policy: str = "lfu",
        predictor: str = "first_k",
        predictor_artifacts: str = "",
        budget_safety: float = 0.9,
        eviction_distance_weight: float = 0.0,
        eviction_freq_decay: float = 1.0,
        eviction_policy_prefill: str | None = None,
        eviction_distance_weight_prefill: float | None = None,
        offload_params: set[str] | None = None,
    ):
        if waves not in (1, 2):
            raise ValueError(f"expert-cache waves must be 1 or 2, got {waves}")
        # Fraction of the estimated copy-free window (last-wave compute + the
        # inter-layer gap, see `_transfer_budget`) actually trusted for
        # prefetch; `1 - budget_safety` is held back as an unused buffer
        # against the estimate being wrong. Swept empirically (exp16/exp17):
        # low values (~0.1-0.3) leave the prefetch channel almost entirely
        # closed; ~0.7-0.9 opens it fully, where the `hybrid` predictor pulls
        # decisively ahead of `first_k` on both hit_rate and latency.
        self._budget_safety = budget_safety
        self.pool_experts = pool_experts
        self.waves = waves
        self.predict_k = predict_k
        self.prefetch_horizon = max(0, prefetch_horizon)
        self.budget_gb = budget_gb
        self.predictor = create_predictor(predictor, artifacts_dir=predictor_artifacts)
        self.eviction = create_eviction(
            eviction_policy, distance_weight=eviction_distance_weight
        )
        # P7.8: only built when explicitly configured, so `_GlobalExpertCache`
        # gets `None` (reuse `self.eviction` for prefill too) by default.
        self.eviction_prefill = (
            create_eviction(
                eviction_policy_prefill,
                distance_weight=(
                    eviction_distance_weight
                    if eviction_distance_weight_prefill is None
                    else eviction_distance_weight_prefill
                ),
            )
            if eviction_policy_prefill is not None
            else None
        )
        self._eviction_freq_decay = eviction_freq_decay
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
        # Per-layer FIFO of not-yet-resolved compute-timing samples. A deque
        # (not a single slot) so a sample that isn't ready yet is retried next
        # time, never silently discarded -- CUDA events recorded on the same
        # stream complete in program order, so checking only the oldest
        # pending entry per layer is sufficient and correct. Each sample's
        # event list has one (start, end) pair per wave -- under waves=2 this
        # excludes the wait for the miss-copy between wave 0 and wave 1, so
        # only genuine compute time is summed (see `iter_expert_waves`).
        self._pending_timing: dict[str, deque[tuple[str, int, list]]] = {}

        # In-flight prefetch batches not yet known-complete: (event, keys).
        # P7.9 -- a FIFO (mirroring `_pending_timing`'s never-discard design)
        # instead of a single slot, so a batch this step doesn't need is not
        # force-waited on and not silently dropped either; see
        # `_wait_for_pending_prefetch`.
        self._pending_prefetch: deque[tuple["torch.cuda.Event", frozenset]] = deque()

        # P7.12: free-list of reusable wave-compute timing Events (the
        # highest-volume Event-creation site: up to 2/wave, 2 waves/layer,
        # 48 layers). Populated by `_release_pooled_events` once a
        # `_pending_timing` entry holding them is popped (fully consumed).
        self._event_pool: list["torch.cuda.Event"] = []

        # Timing-accuracy diagnostics (2026-07-24): how many compute-timing
        # samples actually fed the EMA (`_timing_resolved_count`), and how
        # often the old single-slot discard-on-overwrite design would have
        # silently thrown one away rather than retrying it
        # (`_timing_would_have_discarded_count`) -- see `_consume_timing` and
        # its write site in `iter_expert_waves`.
        self._timing_resolved_count: dict[str, int] = {"prefill": 0, "decode": 0}
        self._timing_would_have_discarded_count: dict[str, int] = {
            "prefill": 0,
            "decode": 0,
        }

        # Inter-layer gap: the transfer budget only ever counted THIS layer's
        # own compute time as available slack, ignoring the (much larger)
        # window between "this layer's prefetch decision" and "the next
        # layer actually needs its experts" -- attention, norms, and any
        # other non-MoE work in between. Measured the same never-discard way
        # as compute timing, via one CUDA event pair spanning consecutive
        # `iter_expert_waves` calls, and folded into `_transfer_budget` so
        # that real (measured) idle copy-engine time unlocks more prefetch.
        self._layer_gap_ema: dict[str, float | None] = {"prefill": None, "decode": None}
        self._pending_gap_timing: deque[tuple] = deque()
        self._last_step_done_ev: "torch.cuda.Event | None" = None
        self._last_step_bucket: str | None = None
        self._gap_resolved_count: dict[str, int] = {"prefill": 0, "decode": 0}

        # Last-wave compute EMA: the copy engine is free for prefetch only
        # AFTER this layer's reactive misses are resident, which (in every
        # branch of `iter_expert_waves`) is exactly when the LAST wave starts
        # -- `wait_event(miss_event)` always precedes it. So the real free
        # window is `last_wave_compute + gap`, and `_transfer_budget` sizes
        # prefetch to that instead of `total_compute - t_miss`. This both is
        # more accurate (no fragile t_miss cancellation) and bounds prefetch
        # so it finishes before the NEXT layer's misses contend for the
        # single shared copy stream.
        self._last_wave_time_ema: dict[str, float | None] = {
            "prefill": None,
            "decode": None,
        }

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

        # Per-bucket EMA of the observed true working-set size
        # (`len(unique(topk_ids))`), fed once per layer-step. Used by
        # `_effective_predict_k`'s auto sizing once warmed, in place of the
        # raw `num_tokens * top_k` upper bound -- which overshoots badly at
        # high batch (measured 64 proposed vs 36.2 truly needed at batch=8)
        # and, per the Phase 7 P7.1 measurement, drives a real ITL/TPOT
        # regression from moving far more experts through the cache than
        # actually needed.
        self._working_set_ema: dict[str, float | None] = {
            "prefill": None,
            "decode": None,
        }

        # Prefetch-budget diagnostics: WHY did prefetch_loaded end up where it
        # did this step? Disambiguates "the predictor/queue ran out of
        # not-already-resident candidates" (queue_exhausted -- more transfer
        # budget would NOT have helped) from "the transfer-budget ceiling was
        # hit while candidates remained" (budget_limited -- more budget WOULD
        # have let more prefetch through). budget_zero is the separate case of
        # _transfer_budget computing no slack at all (no candidates even
        # considered). budget_sum/budget_count let get_stats() report the
        # mean computed budget (in experts) when it was positive.
        self.prefetch_budget_zero: dict[str, int] = {"prefill": 0, "decode": 0}
        self.prefetch_queue_exhausted: dict[str, int] = {"prefill": 0, "decode": 0}
        self.prefetch_budget_limited: dict[str, int] = {"prefill": 0, "decode": 0}
        self.prefetch_budget_sum: dict[str, int] = {"prefill": 0, "decode": 0}
        self.prefetch_budget_count: dict[str, int] = {"prefill": 0, "decode": 0}

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
                    freq_decay=self._eviction_freq_decay,
                    eviction_prefill=self.eviction_prefill,
                )
            elif shapes != expert_shapes:
                raise NotImplementedError(
                    "ExpertCacheOffloader requires homogeneous expert shapes "
                    f"across MoE layers; layer {layer_name!r} has {shapes}, "
                    f"expected {expert_shapes}."
                )

            cache.register_layer(
                layer_name, routed_experts, cpu_storages, self._layer_index[layer_name]
            )

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
        """Pre-load architecturally always-activated routed experts (empty on
        a load-balanced router; hook returns [] by default). This only seeds
        initial residency -- it grants no eviction protection beyond the
        current step's working set, same as any other resident expert."""
        assert self._cache is not None
        for layer_name in self._layer_order:
            always_on = self._always_on_experts(layer_name)
            if not always_on:
                continue
            keys = [(layer_name, g) for g in always_on]
            self._cache.load(keys, horizon=set())

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
            "(waves=%d, predict_k=%s, horizon=%d, eviction=%s, "
            "eviction_prefill=%s, predictor=%s, "
            "PCIe~%.3f ms/expert). Full expert weights in CPU pinned memory: "
            "%.4f GB. GPU-resident pool: %.4f GB.",
            self.pool_experts,
            len(self._layer_order),
            self.waves,
            "auto" if self.predict_k < 0 else self.predict_k,
            self.prefetch_horizon,
            type(self.eviction).__name__,
            (
                type(self.eviction_prefill).__name__
                if self.eviction_prefill is not None
                else "(same)"
            ),
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
        forward context). Prefers `max_query_len` (> 1 means at least one
        request in this batch is multi-token, i.e. a prefill/prefill-chunk;
        pure decode is query_len==1 per request) over `num_prefill_tokens`:
        on FlashAttentionMetadata (this project's default backend) that
        field is only populated under decode-context-parallelism and
        otherwise silently stays 0, which reads as "no prefill" and
        misclassified every step as decode."""
        num_tokens = topk_ids.shape[0]
        try:
            from vllm.forward_context import get_forward_context

            md = get_forward_context().attn_metadata
            if isinstance(md, (list, tuple)):
                md = md[0] if md else None
            if isinstance(md, dict) and md:
                md = next(iter(md.values()))
            mql = getattr(md, "max_query_len", None)
            if mql is not None:
                return "prefill" if mql > 1 else "decode"
            npt = getattr(md, "num_prefill_tokens", None)
            if npt is not None:
                return "prefill" if npt > 0 else "decode"
            nr = getattr(md, "num_reqs", None)
            if nr is not None:
                return "prefill" if num_tokens > nr else "decode"
        except Exception:
            pass
        return "decode" if num_tokens <= 1 else "prefill"

    def _upcoming_layers(self, cur_idx: int) -> list[int]:
        """Layer indices to predict for, up to `prefetch_horizon` steps ahead
        of `cur_idx`, treating layer order as cyclic: the last layer's wave
        can predict the *next* forward pass's early layers, since decode runs
        layers `0..N-1` back-to-back across consecutive token steps with no
        gap between the last layer of one step and the first of the next.
        Capped at `N-1` steps so it never wraps back around to `cur_idx`
        itself (and returns nothing for a single-layer model)."""
        n = len(self._layer_order)
        horizon = min(self.prefetch_horizon, max(0, n - 1))
        return [(cur_idx + offset) % n for offset in range(1, horizon + 1)]

    def _update_working_set_ema(self, bucket: str, n: int) -> None:
        prev = self._working_set_ema[bucket]
        self._working_set_ema[bucket] = (
            float(n)
            if prev is None
            else (1 - self._EMA_ALPHA) * prev + self._EMA_ALPHA * n
        )

    def _effective_predict_k(self, num_tokens: int, top_k: int, bucket: str) -> int:
        """Resolve `self.predict_k` to the k actually queried this step.

        `-1` (auto, default): size to the per-bucket EMA of the *observed*
        true working-set size (`len(unique(topk_ids))`, tracked by
        `_update_working_set_ema`) once warmed, capped at `global_num_experts`
        as a hard ceiling. Falls back to the raw upper bound
        (`num_tokens * top_k`) before the EMA has a first sample. The raw
        bound alone overshoots badly at high batch (measured 64 proposed vs
        36.2 truly needed at batch=8) -- harmless for prediction accuracy
        (the horizon/queue just gets some already-resident padding) but not
        harmless for cost: the Phase 7 P7.1 measurement found it drove a real
        ITL/TPOT regression at batch>=4 from moving several times more
        experts through eviction-protection and prefetch than the step
        actually needed. Tracking the observed size instead closes most of
        that gap while preserving the batch-awareness that made batch=2
        (where the two bounds coincide) a clean win. `0` disables prediction
        outright; a positive int is a static override, mainly for ablation.
        """
        if self.predict_k >= 0:
            return self.predict_k
        raw_bound = num_tokens * top_k
        ema = self._working_set_ema.get(bucket)
        want = raw_bound if ema is None else min(raw_bound, round(ema))
        cap = self._cache.global_num_experts if self._cache is not None else None
        return min(cap, want) if cap is not None else want

    def _predict_upcoming(
        self, cur_idx: int, k: int, bucket: str = "decode"
    ) -> list[Key]:
        """Query the predictor for the upcoming layers ONCE per step, returned
        in priority order (nearest layer first, predictor order within a
        layer) so `_issue_prefetch` can truncate to the transfer budget
        meaningfully. Both consumers -- the eviction-protection horizon
        (`set(...)`) and the prefetch queue (`deque(...)`) -- derive from this
        single call, so a costly predictor (e.g. the hybrid MLP) runs at most
        once per (layer, step) rather than once each in `_horizon_keys` and
        `_issue_prefetch`.

        P7.3 (prefill stream-ahead): during a `bucket == "prefill"` step, a
        chunk touches nearly every expert of the upcoming layer anyway (prefill
        hit_rate measured ~0.38-0.58), so predicting a *subset* is close to
        worthless and the MLP forward is pure waste on a large token count
        (this also delivers P7.10's benefit -- the predictor is never invoked
        during prefill). Bypass the predictor entirely and enqueue *every*
        expert of each upcoming layer instead; `_transfer_budget`'s prefill
        bucket already sizes to the much larger prefill compute window (once
        warmed), so the actual amount fetched is still budget-limited, not
        request-limited. `predict_k == 0` (prediction explicitly disabled)
        still means "predict nothing", even in prefill.
        """
        keys: list[Key] = []
        use_full_range = (
            bucket == "prefill" and self.predict_k != 0 and self._cache is not None
        )
        for tl in self._upcoming_layers(cur_idx):
            lname = self._layer_order[tl]
            ids = (
                range(self._cache.global_num_experts)
                if use_full_range
                else self.predictor.predict(tl, k)
            )
            for g in ids:
                keys.append((lname, g))
        return keys

    def _horizon_keys(self, cur_idx: int, k: int | None = None) -> set[Key]:
        if k is None:
            # No token-count context here to resolve "auto" (-1); only a
            # concrete predict_k is meaningful without it.
            k = self.predict_k if self.predict_k >= 0 else 0
        return set(self._predict_upcoming(cur_idx, k))

    def _effective_t_per_expert_ms(self) -> float:
        """The transfer-cost estimate `_transfer_budget` actually uses.

        Prefers, in order: (1) the P7.5 fitted marginal rate
        (`_transfer_marginal_ms_ema`, the slope of an `elapsed = fixed +
        marginal * n` regression over real transfer-timing samples) -- plain
        `elapsed / n` (what `real_ms_per_expert_ema` computes) folds fixed
        per-batch overhead into the marginal rate, systematically
        overestimating it on the small batches this cache usually transfers,
        which under-sizes the budget and keeps batches small (a
        self-reinforcing pessimism loop); (2) the live EMA of real H2D
        transfer time (`real_ms_per_expert_ema`) once it has a sample but
        before the regression is well-conditioned; (3) the once-at-startup
        `_calibrate_pcie` snapshot. Measured (2026-07-24, exp12): the startup
        calibration was ~16% slower than reality (0.446 vs 0.373 ms/expert
        measured live), so falling through to the live EMA already removes a
        systematic source of overly conservative budgeting; the regression
        removes a further, similarly-shaped bias in the EMA itself."""
        cache = self._cache
        if cache is not None and cache._transfer_marginal_ms_ema is not None:
            return cache._transfer_marginal_ms_ema
        if cache is not None and cache.real_ms_per_expert_ema is not None:
            return cache.real_ms_per_expert_ema
        return self._t_per_expert_ms

    def _transfer_budget(self, bucket: str, num_tokens: int, num_misses: int) -> int:
        """Size speculative prefetch to the copy-engine-free window.

        That window is `last_wave_compute + inter_layer_gap`: the reactive
        misses are already resident before the last wave starts (every branch
        of `iter_expert_waves` does `wait_event(miss_event)` before it), so
        the copy stream is idle for exactly the last wave's compute plus the
        gap until the next layer's router output. No `num_misses` term is
        subtracted -- the miss-loading window is excluded by construction, not
        by arithmetic -- which also stops this layer's prefetch from being
        sized to spill past the gap and contend with the next layer's misses
        on the single shared copy stream. `num_misses` is retained for the
        caller/diagnostics but no longer used here.
        """
        t_per_expert_ms = self._effective_t_per_expert_ms()
        if t_per_expert_ms <= 0:
            return 0
        # Fall back to the total-compute EMA until the last-wave EMA warms up;
        # `_layer_time_ema is None` is the "not warmed at all" gate.
        if self._layer_time_ema[bucket] is None:
            return 0  # not warmed yet: no speculation until we can size it
        last_wave = self._last_wave_time_ema[bucket]
        if last_wave is None:
            last_wave = self._layer_time_ema[bucket]
        gap_ms = self._layer_gap_ema[bucket] or 0.0
        window_ms = (last_wave * num_tokens + gap_ms) * self._budget_safety
        if window_ms <= 0:
            return 0
        return int(window_ms / t_per_expert_ms)

    def _consume_timing(self, layer: str) -> None:
        pending = self._pending_timing.get(layer)
        if not pending:
            return
        while pending:
            bucket, num_tokens, events = pending[0]
            if not events[-1][1].query():
                break  # not ready yet; leave it queued, retry next time
            pending.popleft()
            elapsed = sum(start.elapsed_time(end) for start, end in events)
            per_tok = elapsed / max(1, num_tokens)
            prev = self._layer_time_ema[bucket]
            self._layer_time_ema[bucket] = (
                per_tok
                if prev is None
                else (1 - self._EMA_ALPHA) * prev + self._EMA_ALPHA * per_tok
            )
            # The last wave is the one whose compute overlaps the copy-free
            # window (it starts only after this layer's misses are resident).
            last_start, last_end = events[-1]
            last_per_tok = last_start.elapsed_time(last_end) / max(1, num_tokens)
            last_prev = self._last_wave_time_ema[bucket]
            self._last_wave_time_ema[bucket] = (
                last_per_tok
                if last_prev is None
                else (1 - self._EMA_ALPHA) * last_prev
                + self._EMA_ALPHA * last_per_tok
            )
            self._timing_resolved_count[bucket] += 1
            self._release_pooled_events(events)

    def _get_pooled_event(self) -> "torch.cuda.Event":
        """P7.12: pop a reusable timing-enabled Event instead of allocating a
        fresh one for every wave (`_wave_events` creates up to 2 per wave,
        up to 2 waves per layer, 48 layers -- the highest-volume Event site
        in this class). Safe to reuse only once a pending `_pending_timing`
        entry holding it has been popped (its `elapsed_time()` already read),
        which is exactly when `_release_pooled_events` returns it here."""
        if self._event_pool:
            return self._event_pool.pop()
        return torch.cuda.Event(enable_timing=True)

    def _release_pooled_events(self, events: list[tuple]) -> None:
        for start, end in events:
            if len(self._event_pool) < self._MAX_EVENT_POOL:
                self._event_pool.append(start)
            if len(self._event_pool) < self._MAX_EVENT_POOL:
                self._event_pool.append(end)

    def _consume_gap_timing(self) -> None:
        pending = self._pending_gap_timing
        while pending:
            bucket, start, end = pending[0]
            if not end.query():
                break  # not ready yet; leave it queued, retry next time
            pending.popleft()
            elapsed = start.elapsed_time(end)
            prev = self._layer_gap_ema[bucket]
            self._layer_gap_ema[bucket] = (
                elapsed
                if prev is None
                else (1 - self._EMA_ALPHA) * prev + self._EMA_ALPHA * elapsed
            )
            self._gap_resolved_count[bucket] += 1

    def _wait_for_pending_prefetch(self, needed_keys: set[Key]) -> None:
        """Wait only for in-flight prefetch batches this step will actually
        publish (read via a main-stream D2D copy out of the pool) -- not a
        blanket wait on every previous prefetch (P7.9). Pending batches are a
        FIFO (mirroring `_pending_timing`'s never-discard design): copy-stream
        ops complete in program order, so waiting on the newest intersecting
        entry also guarantees every earlier-enqueued entry is done, and all of
        them (through that index) can be dropped. Entries confirmed complete
        via `event.query()` are dropped opportunistically first, with no wait
        cost, keeping the deque from growing across steps that consistently
        miss."""
        pending = self._pending_prefetch
        while pending and pending[0][0].query():
            pending.popleft()
        if not pending:
            return
        match_idx = -1
        for i, (_, keys) in enumerate(pending):
            if keys & needed_keys:
                match_idx = i
        if match_idx < 0:
            return
        torch.cuda.current_stream().wait_event(pending[match_idx][0])
        for _ in range(match_idx + 1):
            pending.popleft()

    def iter_expert_waves(
        self,
        routed_experts: nn.Module,
        topk_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> Generator[None, None, None]:
        cache = self._cache
        layer = routed_experts.layer_name
        if cache is None or layer not in self._layer_index:
            yield
            return

        cuda = cache._cuda
        self._consume_timing(layer)
        self._consume_gap_timing()
        if cuda:
            cache.drain_transfer_timing()

        if cuda and self._last_step_done_ev is not None:
            gap_end_ev = torch.cuda.Event(enable_timing=True)
            gap_end_ev.record()
            self._pending_gap_timing.append(
                (self._last_step_bucket, self._last_step_done_ev, gap_end_ev)
            )
            self._last_step_done_ev = None

        cur_idx = self._layer_index[layer]
        bucket = self._batch_bucket(topk_ids)
        num_tokens = topk_ids.shape[0]
        top_k = topk_ids.shape[1] if topk_ids.dim() > 1 else 1

        # P7.11: launch the predictor (and `torch.unique`) before
        # materializing `needed` to a CPU list, so one sync captures both
        # instead of two separate serialized ones. `observe()` only needs
        # `hidden_states` (already in scope); the predictor call only needs
        # `predict_k`/`bucket`/`cur_idx` -- neither depends on `needed`.
        # CUDA ops on one stream execute in program order, so the
        # predictor's own sync (the `.tolist()` inside
        # `HybridExpertPredictor.predict`, when it runs one at all --
        # bypassed entirely in prefill by P7.3, and a no-op for `first_k`)
        # transitively waits for the just-queued `unique()` kernel too;
        # reading `needed` back afterward is then close to free rather than
        # a second, separate wait. Uses last step's working-set EMA
        # (one-step-stale, as any EMA read inherently is) since this step's
        # `len(needed)` isn't known yet at this point.
        self.predictor.observe(cur_idx, topk_ids, hidden_states)
        predict_k = self._effective_predict_k(num_tokens, top_k, bucket)
        needed_gpu = torch.unique(topk_ids)
        predicted = self._predict_upcoming(cur_idx, predict_k, bucket)
        horizon = set(predicted)

        needed = needed_gpu.tolist()
        self._update_working_set_ema(bucket, len(needed))
        if cuda:
            # P7.9: only wait for in-flight prefetch batches this step will
            # actually publish (read via a main-stream D2D copy) -- not a
            # blanket wait on every previous prefetch, which stalled the
            # *entire* next layer even when it touched none of the
            # prefetched slots.
            self._wait_for_pending_prefetch({(layer, g) for g in needed})
        resident = {g for g in needed if cache.is_resident(layer, g)}
        hits, misses = split_hits_misses(resident, needed)
        cache.bump(layer, needed)
        self.hot_hits += len(hits)
        self.cold_streamed += len(misses)
        self.bucket_stats[bucket]["hits"] += len(hits)
        self.bucket_stats[bucket]["misses"] += len(misses)

        # Reserve the whole current-iteration working set before any eviction.
        cache.pin_step({(layer, g) for g in needed})
        miss_event = cache.load(
            [(layer, g) for g in misses], horizon, cur_idx, bucket
        )
        self.bytes_streamed += len(misses) * cache.per_expert_bytes

        # Each wave gets its own (start, end) event pair bracketing only its
        # compute kernel(s). Under waves=2 this deliberately excludes the
        # wait_event for the miss-copy between wave 0 and wave 1 -- that wait
        # is idle time, not compute, and including it would inflate
        # `_layer_time_ema` (it used to, by ~2x whenever a step mixed hits
        # and misses, see git history) and make `_transfer_budget` believe
        # there was far more spare time than genuinely existed.
        compute_events: list[tuple] = []

        def _wave_events() -> tuple:
            if not cuda:
                return None, None
            return (self._get_pooled_event(), self._get_pooled_event())  # P7.12

        if self.waves == 1 or not hits or not misses:
            # Single-wave path: waves=1 (stall), no hits (everything must be
            # fetched, a wave-0-over-hits would be empty), or no misses
            # (everything already resident, a wave-1-over-misses would
            # publish nothing and compute a redundant all-zero contribution)
            # -- all three run one kernel over `needed` (== hits when
            # misses is empty). `miss_event` is always None when misses is
            # empty (`_GlobalExpertCache.load` short-circuits on no keys to
            # load), so the wait below is already a no-op in that case.
            if cuda and miss_event is not None:
                torch.cuda.current_stream().wait_event(miss_event)
            start_ev, end_ev = _wave_events()
            cache.publish(layer, needed)
            if start_ev is not None:
                start_ev.record()
            self.wave_count += 1
            yield
            if end_ev is not None:
                end_ev.record()
                compute_events.append((start_ev, end_ev))
        else:
            start_ev0, end_ev0 = _wave_events()
            cache.publish(layer, hits)
            if start_ev0 is not None:
                start_ev0.record()
            self.wave_count += 1
            yield  # wave 0 (hits) overlaps the miss copy
            if end_ev0 is not None:
                end_ev0.record()
                compute_events.append((start_ev0, end_ev0))
            if cuda and miss_event is not None:
                torch.cuda.current_stream().wait_event(miss_event)
            start_ev1, end_ev1 = _wave_events()
            cache.publish(layer, misses)
            if start_ev1 is not None:
                start_ev1.record()
            self.wave_count += 1
            yield
            if end_ev1 is not None:
                end_ev1.record()
                compute_events.append((start_ev1, end_ev1))

        if compute_events:
            pending = self._pending_timing.setdefault(layer, deque())
            if pending:
                # Under the old single-slot design (a plain dict entry
                # overwritten each call) this in-flight sample would have
                # been silently discarded rather than retried next time.
                self._timing_would_have_discarded_count[bucket] += 1
            pending.append((bucket, num_tokens, compute_events))

        self._issue_prefetch(bucket, num_tokens, len(misses), predicted, cur_idx)
        if cuda:
            done_ev = torch.cuda.Event(enable_timing=True)
            done_ev.record()
            self._last_step_done_ev = done_ev
            self._last_step_bucket = bucket
        cache.unpin_step()
        self._maybe_log_stats(cur_idx)

    def _issue_prefetch(
        self,
        bucket: str,
        num_tokens: int,
        num_misses: int,
        predicted: list[Key],
        cur_layer_idx: int,
    ) -> None:
        assert self._cache is not None
        budget = self._transfer_budget(bucket, num_tokens, num_misses)
        if budget <= 0 or self.prefetch_horizon == 0:
            self.prefetch_budget_zero[bucket] += 1
            return
        self.prefetch_budget_sum[bucket] += budget
        self.prefetch_budget_count[bucket] += 1
        # `predicted` is the single per-step predictor query (priority order,
        # upcoming layers only -- current/past layers are stale by construction
        # and never enqueued; cyclic near the end of a pass, see
        # `_upcoming_layers`). Its set is the eviction-protection horizon.
        horizon = set(predicted)
        queue: deque[Key] = deque(predicted)
        batch: list[Key] = []
        while queue and len(batch) < budget:
            key = queue.popleft()
            if self._cache.is_resident(*key):  # promote-on-hit / already there
                continue
            batch.append(key)
        # Disambiguate WHY the loop stopped: candidates ran out first (more
        # transfer budget would not have helped this step) vs the budget
        # ceiling was hit while candidates remained (more budget WOULD help).
        if queue:
            self.prefetch_budget_limited[bucket] += 1
        else:
            self.prefetch_queue_exhausted[bucket] += 1
        if not batch:
            return
        ev = self._cache.load(batch, horizon, cur_layer_idx, bucket)
        self.prefetch_loaded += len(batch)
        self.bucket_stats[bucket]["prefetch_loaded"] += len(batch)
        self.bytes_streamed += len(batch) * self._cache.per_expert_bytes
        if ev is not None:
            self._pending_prefetch.append((ev, frozenset(batch)))

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

    def _budget_diagnostics(self, bucket: str) -> dict[str, float]:
        """Why prefetch stayed low this run: `budget_zero` (no slack was ever
        computed -- more transfer budget headroom wouldn't exist to use),
        `queue_exhausted` (candidates ran out before the budget did -- more
        budget would NOT help), `budget_limited` (the budget ceiling was hit
        while candidates remained -- more budget WOULD help), and the mean
        computed `budget` (in experts) across the calls where it was positive.
        """
        zero = self.prefetch_budget_zero[bucket]
        exhausted = self.prefetch_queue_exhausted[bucket]
        limited = self.prefetch_budget_limited[bucket]
        count = self.prefetch_budget_count[bucket]
        return {
            "budget_zero": zero,
            "queue_exhausted": exhausted,
            "budget_limited": limited,
            "avg_budget": (
                self.prefetch_budget_sum[bucket] / count if count else 0.0
            ),
        }

    def _timing_diagnostics(self, bucket: str) -> dict[str, float]:
        """Accuracy diagnostics for the compute-time input to
        `_transfer_budget`: how many samples resolved and fed the EMA, how
        often the OLD single-slot design would have silently discarded an
        in-flight sample instead of retrying it (see `_consume_timing` and
        its write site), and the EMA's current value."""
        resolved = self._timing_resolved_count[bucket]
        would_discard = self._timing_would_have_discarded_count[bucket]
        return {
            "resolved": resolved,
            "would_have_discarded": would_discard,
            "would_have_discarded_rate": (
                would_discard / resolved if resolved else 0.0
            ),
            "current_ema_ms_per_token": self._layer_time_ema[bucket] or 0.0,
            "current_last_wave_ema_ms_per_token": (
                self._last_wave_time_ema[bucket] or 0.0
            ),
            "gap_resolved": self._gap_resolved_count[bucket],
            "current_gap_ema_ms": self._layer_gap_ema[bucket] or 0.0,
        }

    def _transfer_calibration_stats(self) -> dict[str, float]:
        """Accuracy diagnostics for the transfer-time input to
        `_transfer_budget`: the once-at-startup PCIe calibration, the live
        lifetime average, the P7.5 fitted fixed/marginal regression terms,
        and the value `_transfer_budget` actually uses (see
        `_effective_t_per_expert_ms`) -- these can differ since the EMA
        weights recent samples more than the lifetime average does, and the
        regression's marginal term further removes the fixed-cost bias
        `elapsed / n` folds into the plain EMA."""
        cache = self._cache
        real_count = cache.real_transfer_expert_count if cache else 0
        return {
            "startup_ms_per_expert": self._t_per_expert_ms,
            "real_avg_ms_per_expert": (
                cache.real_transfer_ms_sum / real_count if cache and real_count else 0.0
            ),
            "fitted_marginal_ms_per_expert": (
                cache._transfer_marginal_ms_ema if cache else None
            )
            or 0.0,
            "fitted_fixed_ms": (cache._transfer_fixed_ms_ema if cache else None) or 0.0,
            "effective_ms_per_expert": self._effective_t_per_expert_ms(),
            "real_sample_experts": real_count,
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
            "prefill": {
                **self._bucket_metrics("prefill"),
                **self._budget_diagnostics("prefill"),
                **self._timing_diagnostics("prefill"),
            },
            "decode": {
                **self._bucket_metrics("decode"),
                **self._budget_diagnostics("decode"),
                **self._timing_diagnostics("decode"),
            },
            "transfer_calibration": self._transfer_calibration_stats(),
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
        dbg = self._budget_diagnostics("decode")
        logger.info(
            "[ExpertCacheOffloader] prefetch budget (pass=%d) decode: "
            "budget_zero=%d queue_exhausted=%d budget_limited=%d avg_budget=%.2f",
            self._pass_count,
            dbg["budget_zero"], dbg["queue_exhausted"], dbg["budget_limited"],
            dbg["avg_budget"],
        )
        td = self._timing_diagnostics("decode")
        xfer = self._transfer_calibration_stats()
        logger.info(
            "[ExpertCacheOffloader] timing accuracy (pass=%d) decode: "
            "resolved=%d would_have_discarded=%d (rate=%.3f) "
            "ema_ms_per_tok=%.4f last_wave_ms_per_tok=%.4f "
            "gap_ema_ms=%.4f (gap_resolved=%d) | "
            "transfer: startup_ms_per_expert=%.4f "
            "real_avg_ms_per_expert=%.4f fitted_marginal_ms_per_expert=%.4f "
            "fitted_fixed_ms=%.4f effective_ms_per_expert=%.4f (n=%d)",
            self._pass_count,
            td["resolved"], td["would_have_discarded"],
            td["would_have_discarded_rate"], td["current_ema_ms_per_token"],
            td["current_last_wave_ema_ms_per_token"],
            td["current_gap_ema_ms"], td["gap_resolved"],
            xfer["startup_ms_per_expert"], xfer["real_avg_ms_per_expert"],
            xfer["fitted_marginal_ms_per_expert"], xfer["fitted_fixed_ms"],
            xfer["effective_ms_per_expert"], xfer["real_sample_experts"],
        )

    def _log_final_stats(self) -> None:
        # Registered via atexit; never let shutdown-time logging raise.
        try:
            logger.info("[ExpertCacheOffloader] final stats:")
            self._log_stats_line()
        except Exception:
            pass
