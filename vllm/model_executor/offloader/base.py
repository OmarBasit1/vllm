# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

if TYPE_CHECKING:
    from vllm.config import OffloadConfig

logger = init_logger(__name__)


def should_pin_memory() -> bool:
    """Check if pinned memory should be used for weight offloading.

    Combines the platform capability check with the user override env var.
    On unified-memory systems (e.g. GH200) pinned memory eats into GPU
    memory, so users can disable it via VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY.
    """
    return (
        is_pin_memory_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
    )


"""
class relation:

BaseOffloader (ABC)
  * implemented by: UVAOffloader
  * implemented by: PrefetchOffloader
    * uses: _ModuleOffloader
        * uses: _BaseParamOffloader (ABC)
            * implemented by: _CpuParamOffloader
"""


class BaseOffloader(ABC):
    """Base class for model parameter offloading strategies.

    Offloaders control how model parameters are stored and loaded during
    inference. Different strategies trade memory for compute/transfer time.
    """

    @abstractmethod
    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.

        Returns:
            List of modules, potentially with offloading hooks installed.
        """
        pass

    def post_init(self):
        """Called after model construction completes.

        Offloaders can use this to:
        - Finalize parameter storage
        - Start initial prefetching
        - Allocate shared resources
        """
        return

    def sync_prev_onload(self) -> None:  # noqa: B027
        """Sync previous onload operations. Override in subclasses."""
        pass

    def join_after_forward(self) -> None:  # noqa: B027
        """Join streams after forward. Override in subclasses."""
        pass

    def _wait_for_layer(self, layer_idx: int) -> None:  # noqa: B027
        """Wait for layer prefetch. Override in subclasses."""
        pass

    def _start_prefetch(self, layer_idx: int) -> None:  # noqa: B027
        """Start layer prefetch. Override in subclasses."""
        pass

    def resident_bytes(self) -> int:
        """GPU-resident weight bytes this offloader allocates in ``post_init``
        (outside the model-loading memory profiler).

        The model runner folds this into ``model_memory_usage`` so KV-cache
        sizing accounts for it. Returns 0 by default (offloaders whose buffers
        are already counted during model loading, or which use pageable/managed
        memory). Override in subclasses that self-allocate large device
        buffers after loading.
        """
        return 0

    def iter_expert_waves(
        self, routed_experts: nn.Module, topk_ids: torch.Tensor
    ) -> Generator[None, None, None]:
        """Yield once per residency "wave" the MoE kernel must run for this
        step, managing GPU residency of ``routed_experts``' experts before each
        yield.

        The runner calls the fused-MoE kernel once per yield and sums the
        outputs, so an offloader can tile a step's expert compute across
        several passes when not all needed experts fit GPU memory at once.
        The default yields exactly once with no residency management, so every
        offloader other than the expert-cache one runs the kernel a single
        time with unchanged behavior. Override in subclasses.
        """
        yield


class NoopOffloader(BaseOffloader):
    """No-op offloader that returns modules as-is without any offloading."""

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance (defaults to no-op).
_instance: BaseOffloader = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    global _instance
    _instance = instance
    if isinstance(instance, NoopOffloader):
        logger.debug_once("Offloader set to NoopOffloader (no offloading).")
    else:
        logger.info_once("Offloader set to %s", type(instance).__name__)


def create_offloader(offload_config: "OffloadConfig") -> BaseOffloader:
    """Create an offloader based on the offload configuration.

    Uses the explicit ``offload_backend`` selector.  When set to ``"auto"``,
    selects expert_cache if ``cache_capacity > 0``, else prefetch if
    ``offload_group_size > 0``, else UVA if ``cpu_offload_gb > 0``, otherwise
    noop.
    """
    from vllm.model_executor.offloader.expert_cache import ExpertCacheOffloader
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    backend = offload_config.offload_backend
    uva = offload_config.uva
    prefetch = offload_config.prefetch
    expert_cache = offload_config.expert_cache

    if backend == "auto":
        if expert_cache.cache_capacity > 0:
            backend = "expert_cache"
        elif prefetch.offload_group_size > 0:
            backend = "prefetch"
        elif uva.cpu_offload_gb > 0:
            backend = "uva"
        else:
            return NoopOffloader()

    if backend == "prefetch":
        return PrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    elif backend == "uva":
        return UVAOffloader(
            cpu_offload_max_bytes=int(uva.cpu_offload_gb * 1024**3),
            cpu_offload_params=uva.cpu_offload_params,
        )
    elif backend == "expert_cache":
        return ExpertCacheOffloader(
            pool_experts=expert_cache.cache_capacity,
            waves=expert_cache.waves,
            predict_k=expert_cache.predict_k,
            prefetch_horizon=expert_cache.prefetch_horizon,
            budget_gb=expert_cache.budget_gb,
            eviction_policy=expert_cache.eviction_policy,
            predictor=expert_cache.predictor,
            offload_params=expert_cache.offload_params,
        )
    else:
        return NoopOffloader()
