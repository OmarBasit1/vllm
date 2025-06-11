# SPDX-License-Identifier: Apache-2.0
import contextlib
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import pynvml
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_nvml_freq_active = False  # Prevent nested or concurrent frequency setting
_nvml_freq_lock = threading.Lock()  # Lock to prevent race conditions


def nvml_get_available_freq():
    """
    Returns a sorted list of available GPU clock frequencies at the highest
    memory clock setting.
    """
    pynvml.nvmlInit()
    handle = _get_gpu_handles()[0]

    memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
    highest_memory_clock = max(memory_clocks)

    return sorted(
        pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle,
                                                    highest_memory_clock))


def uniform_sample_sorted(lst, k):
    """
    Selects `k` elements from the sorted input list as uniformly as possible,
    ensuring the first and last elements are included.
    """
    if k < 2 or k > len(lst):
        raise ValueError(
            "k must be at least 2 and at most the length of the list")
    lst = sorted(lst)
    step = (len(lst) - 1) / (k - 1)
    indices = sorted(set(round(i * step) for i in range(k)))
    return [lst[i] for i in indices]


@contextlib.contextmanager
def nvml_lock_freq(freq):
    """
    Context manager that temporarily locks GPU frequency for all GPUs in
    `CUDA_VISIBLE_DEVICES`. Prevents nested or concurrent usage across threads.
    """
    global _nvml_freq_active

    with _nvml_freq_lock:  # Ensure thread safety
        if _nvml_freq_active:
            raise RuntimeError(
                'nvml_lock_freq is already active in another thread!')

        _nvml_freq_active = True

    handles = _get_gpu_handles()
    try:
        for handle in handles:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
            _retry_nvml_call(pynvml.nvmlDeviceSetGpuLockedClocks, handle, freq,
                             freq)
        logger.info('Locking GPU freq at %d MHz ...', freq)
        yield
    finally:
        for handle in handles:
            _retry_nvml_call(pynvml.nvmlDeviceResetGpuLockedClocks, handle)
        with _nvml_freq_lock:
            _nvml_freq_active = False
        logger.info('Resetting GPU freq ...')


def _retry_nvml_call(fn, *args, retries=3, delay=0.5):
    for attempt in range(retries):
        try:
            fn(*args)
            return
        except pynvml.NVMLError as e:
            logger.warning("NVML call failed with error '%s' (attempt %d/%d)",
                           str(e), attempt + 1, retries)
            if attempt < retries - 1:
                try:
                    pynvml.nvmlShutdown()
                except pynvml.NVMLError as shutdown_err:
                    logger.debug("nvmlShutdown failed: %s", shutdown_err)
                try:
                    pynvml.nvmlInit()
                except pynvml.NVMLError as init_err:
                    logger.debug("nvmlInit failed: %s", init_err)
                time.sleep(delay)
            else:
                raise


def nvml_set_freq(freq):
    """
    Function that sets the GPU frequency for all GPUs in `CUDA_VISIBLE_DEVICES`.
    If the context manager `nvml_lock_freq` is active, raises an exception.
    """
    global _nvml_freq_active

    with _nvml_freq_lock:
        if _nvml_freq_active:
            raise RuntimeError(
                'Cannot set GPU frequency while nvml_lock_freq is active!')

    handles = _get_gpu_handles()

    def set_freq(handle):
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)

    with ThreadPoolExecutor(max_workers=len(handles)) as executor:
        futures = [executor.submit(set_freq, handle) for handle in handles]
        for future in as_completed(futures):
            future.result()  # Will raise if any thread fails

    logger.info('Set GPU freq to %d MHz for all devices.', freq)


def _get_gpu_handles():
    pynvml.nvmlInit()
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        gpu_indices = [int(i) for i in cuda_visible_devices.split(',')]
    else:
        gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))
    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]


@contextmanager
def timeit(name='Unnamed code block'):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info('[%s] execution time: %f', name, elapsed_time)


class CSVWriter:

    def __init__(self, col_names: list[str], filename: Path):
        self.filename = str(filename)
        self.col_names = col_names
        self.header_written = False  # Track if the header has been written

        # ruff: noqa: SIM115
        self.file = open(self.filename, 'a', newline='', encoding='utf-8')
        # ruff: enable
        self.writer = csv.writer(self.file)

    def add_row(self, row: list):
        assert len(row) == len(self.col_names)
        if not self.header_written:
            self.writer.writerow(self.col_names)
            self.header_written = True
        self.writer.writerow(row)

    def close(self):
        self.file.close()


# Removed T4 freqs as their similar performance stops
# greedy MPC to stop looking at lower freqs
# Removed boost freq for A100-TP4 (1410)
# Removed boost freq for H100 (1410)
def get_preselected_freq(gpu: str) -> list[int]:
    return {
        'T4': [300, 450, 585, 735, 870, 1020, 1155, 1305, 1440, 1590],
        'A40': [210, 375, 555, 720, 885, 1065, 1230, 1395, 1575, 1740],
        'A100-SXM4-80GB':
        [210, 345, 480, 615, 750, 870, 1005, 1140, 1275, 1410],
        'H100-80GB-HBM3':
        [345, 525, 705, 885, 1065, 1260, 1440, 1620, 1800, 1980],
    }[gpu]


def get_gpu_name():
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found')
    ret = torch.cuda.get_device_name(0)
    ret = ret.replace('NVIDIA ', '')
    ret = ret.replace('Tesla ', '')
    ret = ret.replace(' ', '-')
    return ret


if __name__ == '__main__':
    freqs = uniform_sample_sorted(nvml_get_available_freq(), 10)
    print(freqs)