# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import shared_memory
from unittest.mock import patch

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# Constants
_TMP_DIR = tempfile.gettempdir()
_LOCK_FILE_PREFIX = os.path.join(_TMP_DIR, "vllm_routed_experts")
_BUFFER_PREFIX = "vllm_routed_experts_buffer"
_PROBABILITY_BUFFER_PREFIX = "vllm_routed_experts_probability_buffer"
_LAYER0_INPUT_BUFFER_PREFIX = "vllm_routed_experts_layer0_input_buffer"

# Global singleton instances
_global_experts_capturer: RoutedExpertsCapturer | None = None
_global_experts_reader: RoutedExpertsReader | None = None


@contextmanager
def _file_lock(lock_file: str, mode: str = "wb+") -> Generator[None, None, None]:
    """Context manager for file-based locking."""
    with open(lock_file, mode) as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)


def _create_or_attach_shared_memory(
    name: str, size: int, lock_file: str
) -> shared_memory.SharedMemory:
    """Create or attach to shared memory with proper locking."""
    # Ensure lock file exists before acquiring lock
    with open(lock_file, "wb"):
        pass

    with _file_lock(lock_file):
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=name, create=False, size=size)

        if shm.size != size:
            logger.warning(
                "Shared memory %s size mismatch; recreating",
                name,
            )
            shm.close()
            shm.unlink()
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                logger.info("Created shared memory %s", name)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False, size=size)
                logger.info("Linked to existing shared memory %s", name)

    return shm


class RoutedExpertsCapturer:
    """
    Capturer for routed experts with device and optional shared memory buffer.

    This class captures expert routing decisions during model forward passes
    and optionally stores them in shared memory for cross-process access.
    """

    _instance: RoutedExpertsCapturer | None = None

    def __init__(self) -> None:
        self._device_buffer: torch.Tensor | None = None
        self._device_probability_buffer: torch.Tensor | None = None
        self._device_layer0_input_buffer: torch.Tensor | None = None
        self._num_experts = 0
        self._hidden_size = 0
        self._shm: shared_memory.SharedMemory | None = None
        self._probability_shm: shared_memory.SharedMemory | None = None
        self._layer0_input_shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None
        self._host_probability_buffer_view: np.ndarray | None = None
        self._host_layer0_input_buffer_view: np.ndarray | None = None
        self._lock_file: str | None = None

    @classmethod
    def create(cls) -> RoutedExpertsCapturer:
        """Create a global singleton instance."""
        global _global_experts_capturer
        if _global_experts_capturer is not None:
            raise RuntimeError("Experts capturer already created.")

        _global_experts_capturer = cls()
        return _global_experts_capturer

    @staticmethod
    def get_instance() -> RoutedExpertsCapturer | None:
        """Get the global singleton instance."""
        return _global_experts_capturer

    def init_buffer(
        self,
        max_num_batched_tokens: int,
        max_num_kv_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        """
        Initialize the device buffer and optionally shared memory buffer.

        Args:
            max_num_batched_tokens: Maximum number of tokens in a batch.
            max_num_kv_tokens: Maximum number of KV tokens for shared memory.
            vllm_config: vllm configuration containing layer and expert info.
        """

        if self._device_buffer is not None:
            raise RuntimeError("Device buffer has already been initialized")

        hf_config = vllm_config.model_config.hf_text_config
        num_layers = hf_config.num_hidden_layers
        num_experts_per_tok = hf_config.num_experts_per_tok
        num_experts = int(vllm_config.model_config.get_num_experts())
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0 for MoE routed expert capture")
        self._num_experts = num_experts
        hidden_size = int(vllm_config.model_config.get_hidden_size())
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0 for MoE routed expert capture")
        self._hidden_size = hidden_size

        # Initialize device buffer
        self._device_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts_per_tok),
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self._device_probability_buffer = torch.zeros(
            (max_num_batched_tokens, num_layers, num_experts),
            dtype=torch.float16,
            device=current_platform.device_type,
        )
        self._device_layer0_input_buffer = torch.zeros(
            (max_num_batched_tokens, hidden_size),
            dtype=torch.float16,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return

        # Initialize shared memory
        shape = (max_num_kv_tokens, num_layers, num_experts_per_tok)
        probability_shape = (max_num_kv_tokens, num_layers, num_experts)
        layer0_input_shape = (max_num_kv_tokens, hidden_size)
        buffer_size = int(np.prod(shape)) * np.dtype(np.int32).itemsize
        probability_buffer_size = (
            int(np.prod(probability_shape)) * np.dtype(np.float16).itemsize
        )
        layer0_input_buffer_size = (
            int(np.prod(layer0_input_shape)) * np.dtype(np.float16).itemsize
        )
        instance_id = vllm_config.instance_id
        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}_{self.dp_rank}.lock"
        shm_name = f"{_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"
        probability_shm_name = (
            f"{_PROBABILITY_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"
        )
        layer0_input_shm_name = (
            f"{_LAYER0_INPUT_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"
        )

        self._shm = _create_or_attach_shared_memory(
            shm_name, buffer_size, self._lock_file
        )
        self._probability_shm = _create_or_attach_shared_memory(
            probability_shm_name, probability_buffer_size, self._lock_file
        )
        self._layer0_input_shm = _create_or_attach_shared_memory(
            layer0_input_shm_name,
            layer0_input_buffer_size,
            self._lock_file,
        )
        self._host_buffer_view = np.ndarray(shape, dtype=np.int32, buffer=self._shm.buf)
        self._host_probability_buffer_view = np.ndarray(
            probability_shape,
            dtype=np.float16,
            buffer=self._probability_shm.buf,
        )
        self._host_layer0_input_buffer_view = np.ndarray(
            layer0_input_shape,
            dtype=np.float16,
            buffer=self._layer0_input_shm.buf,
        )
        self._host_buffer_view.fill(0)
        self._host_probability_buffer_view.fill(0)
        self._host_layer0_input_buffer_view.fill(0)

        logger.debug(
            "Created shared memory buffer '%s' with shape %s",
            shm_name,
            shape,
        )

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        expert_probabilities: torch.Tensor | None = None,
    ) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
            topk_weights: Optional routing weights for selected experts.
            hidden_states: Optional layer input embeddings (batch_size, hidden_size).
            expert_probabilities: Optional full expert probability map
                (batch_size, num_experts).
        """
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")
        if self._device_probability_buffer is None:
            raise RuntimeError(
                "Probability buffer not initialized. Call init_buffer() first."
            )
        if self._device_layer0_input_buffer is None:
            raise RuntimeError(
                "Layer-0 input buffer not initialized. Call init_buffer() first."
            )

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            token_num_per_dp = ctx.dp_metadata.num_tokens_across_dp_cpu[self.dp_rank]
            cumsum = torch.cumsum(ctx.dp_metadata.num_tokens_across_dp_cpu, dim=0)
            assert cumsum[-1] == topk_ids.shape[0]
            end_loc = cumsum[self.dp_rank]
            start_loc = end_loc - token_num_per_dp

        if layer_id >= self._device_buffer.shape[1]:
            return

        topk_ids_for_dp = topk_ids[start_loc:end_loc, :]
        self._device_buffer[:token_num_per_dp, layer_id, :] = topk_ids_for_dp

        if layer_id == 0 and hidden_states is not None:
            layer0_input_for_dp = hidden_states[start_loc:end_loc, :]
            self._device_layer0_input_buffer[:token_num_per_dp, :] = (
                layer0_input_for_dp.to(dtype=self._device_layer0_input_buffer.dtype)
            )

        prob_slice = self._device_probability_buffer[:token_num_per_dp, layer_id, :]
        prob_slice.zero_()

        if expert_probabilities is not None:
            prob_for_dp = expert_probabilities[start_loc:end_loc, :]
            prob_slice.copy_(prob_for_dp.to(dtype=prob_slice.dtype), non_blocking=True)
            return

        topk_ids_long = topk_ids_for_dp.to(dtype=torch.long)
        if topk_weights is None:
            topk_weights_for_dp = torch.ones(
                topk_ids_long.shape,
                dtype=prob_slice.dtype,
                device=topk_ids_long.device,
            )
        else:
            topk_weights_for_dp = topk_weights[start_loc:end_loc, :].to(
                dtype=prob_slice.dtype
            )

        valid_mask = (topk_ids_long >= 0) & (topk_ids_long < self._num_experts)
        safe_ids = torch.where(
            valid_mask,
            topk_ids_long,
            torch.zeros_like(topk_ids_long),
        )
        safe_weights = torch.where(
            valid_mask,
            topk_weights_for_dp,
            torch.zeros_like(topk_weights_for_dp),
        )
        prob_slice.scatter_add_(1, safe_ids, safe_weights)

        row_sum = prob_slice.sum(dim=1, keepdim=True)
        prob_slice.div_(torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum)))

    def clear_buffer(self) -> None:
        """Clear the device buffer."""
        if self._device_buffer is not None:
            self._device_buffer.zero_()
        if self._device_probability_buffer is not None:
            self._device_probability_buffer.zero_()
        if self._device_layer0_input_buffer is not None:
            self._device_layer0_input_buffer.zero_()

    def save_captured_experts(self, indices: np.ndarray) -> None:
        """
        Save captured experts from device buffer to shared memory.

        Args:
            indices: Array of indices indicating where to store the data.
        """
        if get_tensor_model_parallel_rank() != 0:
            return
        if self._lock_file is None:
            raise RuntimeError("Shared memory not initialized.")
        if self._host_buffer_view is None:
            return
        if self._host_probability_buffer_view is None:
            return
        if self._host_layer0_input_buffer_view is None:
            return
        if self._device_buffer is None:
            raise RuntimeError("Device buffer not initialized.")
        if self._device_probability_buffer is None:
            raise RuntimeError("Probability device buffer not initialized.")
        if self._device_layer0_input_buffer is None:
            raise RuntimeError("Layer-0 input device buffer not initialized.")

        num_tokens = len(indices)
        data = self._device_buffer[:num_tokens, :, :].cpu().numpy()
        probability_data = self._device_probability_buffer[
            :num_tokens, :, :
        ].cpu().numpy()
        layer0_input_data = self._device_layer0_input_buffer[
            :num_tokens, :
        ].cpu().numpy()

        with _file_lock(self._lock_file):
            self._host_buffer_view[indices, :, :] = data
            self._host_probability_buffer_view[indices, :, :] = probability_data
            self._host_layer0_input_buffer_view[indices, :] = layer0_input_data

    def cleanup(self) -> None:
        """Explicitly clean up shared memory resources."""
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                logger.debug("Exception during cleanup for capturer", exc_info=True)
            finally:
                self._shm = None
        if self._probability_shm is not None:
            try:
                self._probability_shm.close()
                self._probability_shm.unlink()
            except Exception:
                logger.debug(
                    "Exception during probability cleanup for capturer",
                    exc_info=True,
                )
            finally:
                self._probability_shm = None
        if self._layer0_input_shm is not None:
            try:
                self._layer0_input_shm.close()
                self._layer0_input_shm.unlink()
            except Exception:
                logger.debug(
                    "Exception during layer-0 input cleanup for capturer",
                    exc_info=True,
                )
            finally:
                self._layer0_input_shm = None

    def __del__(self) -> None:
        """Clean up shared memory on destruction."""
        self.cleanup()


class RoutedExpertsReader:
    """
    Reader for routed experts from shared memory.

    This class attaches to shared memory created by RoutedExpertsCapturer
    and reads expert routing decisions.
    """

    _instance: RoutedExpertsReader | None = None

    def __init__(self) -> None:
        self._shm: shared_memory.SharedMemory | None = None
        self._probability_shm: shared_memory.SharedMemory | None = None
        self._layer0_input_shm: shared_memory.SharedMemory | None = None
        self._host_buffer_view: np.ndarray | None = None
        self._host_probability_buffer_view: np.ndarray | None = None
        self._host_layer0_input_buffer_view: np.ndarray | None = None
        self._lock_file: str | None = None

    @classmethod
    def create(cls) -> RoutedExpertsReader:
        """Create a global singleton instance."""
        global _global_experts_reader
        if _global_experts_reader is not None:
            raise RuntimeError("Experts reader already created.")

        _global_experts_reader = cls()
        return _global_experts_reader

    @staticmethod
    def get_instance() -> RoutedExpertsReader | None:
        """Get the global singleton instance."""
        if _global_experts_reader is None:
            logger.info("Experts reader not initialized.")
        return _global_experts_reader

    def attach_buffer(
        self,
        max_num_kv_tokens: int,
        vllm_config: VllmConfig,
    ) -> None:
        """
        Attach to an existing shared memory buffer.

        Args:
            max_num_kv_tokens: Maximum number of KV tokens.
            vllm_config: vllm configuration.
        """
        if self._shm is not None:
            logger.warning("Already attached to shared memory buffer.")
            return  # Already attached

        hf_config = vllm_config.model_config.hf_text_config
        num_experts = int(vllm_config.model_config.get_num_experts())
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0 for MoE routed expert capture")
        hidden_size = int(vllm_config.model_config.get_hidden_size())
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0 for MoE routed expert capture")
        shape = (
            max_num_kv_tokens,
            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
        )
        probability_shape = (
            max_num_kv_tokens,
            hf_config.num_hidden_layers,
            num_experts,
        )
        layer0_input_shape = (max_num_kv_tokens, hidden_size)

        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        instance_id = vllm_config.instance_id
        self._lock_file = f"{_LOCK_FILE_PREFIX}_{instance_id}_{self.dp_rank}.lock"
        shm_name = f"{_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"
        probability_shm_name = (
            f"{_PROBABILITY_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"
        )
        layer0_input_shm_name = (
            f"{_LAYER0_INPUT_BUFFER_PREFIX}_{instance_id}_{self.dp_rank}"
        )

        with _file_lock(self._lock_file, mode="rb+"):
            # Avoid resource_tracker registering the shared memory
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                self._shm = shared_memory.SharedMemory(name=shm_name)
                self._probability_shm = shared_memory.SharedMemory(
                    name=probability_shm_name
                )
                self._layer0_input_shm = shared_memory.SharedMemory(
                    name=layer0_input_shm_name
                )

            self._host_buffer_view = np.ndarray(
                shape, dtype=np.int32, buffer=self._shm.buf
            )
            self._host_probability_buffer_view = np.ndarray(
                probability_shape,
                dtype=np.float16,
                buffer=self._probability_shm.buf,
            )
            self._host_layer0_input_buffer_view = np.ndarray(
                layer0_input_shape,
                dtype=np.float16,
                buffer=self._layer0_input_shm.buf,
            )

    def get_routed_experts(self, indices: np.ndarray) -> np.ndarray:
        """
        Read routed expert data from shared memory.

        Args:
            indices: Array of indices to read.

        Returns:
            Copy of the expert routing data for the given indices.
        """
        if self._host_buffer_view is None:
            raise RuntimeError("Buffer not attached. Call attach_buffer() first.")
        if self._lock_file is None:
            raise RuntimeError("Lock file not initialized.")

        with _file_lock(self._lock_file, mode="rb+"):
            return self._host_buffer_view[indices, :, :].copy()

    def get_routed_expert_probabilities(self, indices: np.ndarray) -> np.ndarray:
        """
        Read routed expert probability maps from shared memory.

        Args:
            indices: Array of indices to read.

        Returns:
            Copy of the expert routing probability maps for the given indices.
        """
        if self._host_probability_buffer_view is None:
            raise RuntimeError(
                "Probability buffer not attached. Call attach_buffer() first."
            )
        if self._lock_file is None:
            raise RuntimeError("Lock file not initialized.")

        with _file_lock(self._lock_file, mode="rb+"):
            return self._host_probability_buffer_view[indices, :, :].copy()

    def get_layer0_input_embeddings(self, indices: np.ndarray) -> np.ndarray:
        """
        Read layer-0 input embeddings from shared memory.

        Args:
            indices: Array of indices to read.

        Returns:
            Copy of the layer-0 input embeddings for the given indices.
        """
        if self._host_layer0_input_buffer_view is None:
            raise RuntimeError(
                "Layer-0 input buffer not attached. Call attach_buffer() first."
            )
        if self._lock_file is None:
            raise RuntimeError("Lock file not initialized.")

        with _file_lock(self._lock_file, mode="rb+"):
            return self._host_layer0_input_buffer_view[indices, :].copy()

    def cleanup(self) -> None:
        """Explicitly clean up resources (close without unlink)."""
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                logger.debug("Exception during cleanup for reader", exc_info=True)
            finally:
                self._shm = None
        if self._probability_shm is not None:
            try:
                self._probability_shm.close()
            except Exception:
                logger.debug(
                    "Exception during probability cleanup for reader",
                    exc_info=True,
                )
            finally:
                self._probability_shm = None
        if self._layer0_input_shm is not None:
            try:
                self._layer0_input_shm.close()
            except Exception:
                logger.debug(
                    "Exception during layer-0 input cleanup for reader",
                    exc_info=True,
                )
            finally:
                self._layer0_input_shm = None

    def __del__(self) -> None:
        """Close shared memory on destruction (do not unlink)."""
        self.cleanup()
