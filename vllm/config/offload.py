# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch", "expert_cache"]
EvictionPolicy = Literal["lfu"]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class ExpertCacheOffloadConfig:
    """Configuration for the global cross-layer expert cache offloader.

    One GPU pool of expert rows is shared across *all* MoE layers (only one runs
    at a time). Experts a step needs that are already resident ("hits") cost no
    transfer; the rest ("misses") are loaded on demand, evicting the
    least-valuable resident experts (graded LFU). Because one layer's full
    working set (`<= global_num_experts`) always fits the pool, a layer is
    computed in a fixed 1 or 2 "waves" rather than `ceil(cold/width)`: `waves=1`
    stalls on all misses then runs one kernel; `waves=2` computes the resident
    hits while the misses stream in, then computes the misses. Predicted
    upcoming-layer experts are prefetched within a measured PCIe budget. Requires
    `enforce_eager` (wave/eviction decisions inspect runtime `topk_ids`) and the
    Triton MoE kernel.
    """

    cache_capacity: int = Field(default=0, ge=0)
    """Total experts the shared GPU pool holds across ALL layers (not per-layer).
    This is the entire GPU-resident footprint. Must be `>= global_num_experts` so
    any single layer's working set fits. Default 0 means disabled."""

    waves: int = Field(default=2)
    """Fixed wave count per layer-step. 1 = stall (load all misses, then one
    kernel over hits+misses). 2 = overlap (compute resident hits while misses
    stream in, then compute misses; the runner sums the two)."""

    max_transient_experts: int = Field(default=0, ge=0)
    """Deprecated no-op, retained only so existing command lines / scripts do not
    break. The old per-wave streaming scratch no longer exists (misses stream
    directly into the shared pool); this value is ignored."""

    predict_k: int = Field(default=0, ge=0)
    """Number of experts the predictor proposes per upcoming layer for
    prefetch. Default 0 disables prefetch prediction."""

    prefetch_horizon: int = Field(default=1, ge=0)
    """How many layers ahead to prefetch predicted experts. 0 disables
    prefetch."""

    budget_gb: float = Field(default=0, ge=0)
    """If > 0, verify the measured GPU-resident pool footprint is within 15% of
    this target and fail fast otherwise. Use it to pin real memory parity with a
    UVA baseline's `--cpu-offload-gb` budget instead of deriving `cache_capacity`
    by hand. Default 0 disables the check."""

    eviction_policy: EvictionPolicy = "lfu"
    """Eviction policy for the shared pool. Only "lfu" is shipped (graded:
    predicted-soon experts are evicted only as a last resort)."""

    predictor: str = "first_k"
    """Predictor identifier. "first_k" (the only one shipped) statically
    predicts global expert ids 0..predict_k-1 for every layer and request,
    with no router-logit awareness."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for expert caching.
    Unmatched parameters are not cached. If this set is empty, ALL per-expert
    parameters of each MoE layer are targeted. Uses the same segment-matching
    semantics as `PrefetchOffloadConfig.offload_params`."""


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (expert_cache if cache_capacity > 0, else prefetch if
      offload_group_size > 0, else uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    - "expert_cache": Global cross-layer expert cache with single-pass
      (selectable 1/2 wave) compute and budgeted prefetch. Requires
      `enforce_eager=True` (see `ExpertCacheOffloadConfig`).
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    expert_cache: ExpertCacheOffloadConfig = Field(
        default_factory=ExpertCacheOffloadConfig
    )
    """Parameters for the expert-cache offloading backend."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        if (
            self.offload_backend == "expert_cache"
            or self.expert_cache.cache_capacity > 0
        ):
            if self.expert_cache.waves not in (1, 2):
                raise ValueError(
                    f"expert_cache.waves ({self.expert_cache.waves}) must be 1 or 2"
                )

        # Warn if multiple backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        expert_cache_active = self.expert_cache.cache_capacity > 0
        # Listed in "auto" priority order (expert_cache > prefetch > uva) so
        # active_backends[0] below is actually the one that gets selected.
        active_backends = [
            name
            for name, active in (
                ("expert_cache", expert_cache_active),
                ("prefetch", prefetch_active),
                ("uva", uva_active),
            )
            if active
        ]
        if self.offload_backend != "auto":
            other_active = [b for b in active_backends if b != self.offload_backend]
            if other_active:
                warnings.warn(
                    f"{other_active} offload fields are set but "
                    f"offload_backend='{self.offload_backend}'. "
                    f"{other_active} settings will be ignored.",
                    stacklevel=2,
                )
        elif len(active_backends) > 1:
            warnings.warn(
                f"Multiple offload backends have fields set "
                f"({active_backends}) with offload_backend='auto'. "
                f"'{active_backends[0]}' will be selected (expert_cache takes "
                f"priority over prefetch, which takes priority over uva). "
                f"Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        return self

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph, and ExpertCacheOffloader mutates weight
        buffers/expert maps in a data-dependent way each step. Changing any
        offload setting can alter which layers are hooked and how prefetch
        or cache-fill decisions are computed, so the compilation cache must
        distinguish them.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
