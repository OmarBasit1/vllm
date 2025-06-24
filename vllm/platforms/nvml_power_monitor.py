# SPDX-License-Identifier: Apache-2.0
import contextlib
import csv
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Optional

import list
import pynvml

from vllm.logger import init_logger

logger = init_logger(__name__)


def get_nvml_metric_value(handle, field_id):
    try:
        metric = pynvml.nvmlDeviceGetFieldValues(handle, [field_id])[0]
        if metric.nvmlReturn != pynvml.NVML_SUCCESS:
            return 0  # Log power as 0 if NVML API returns an error
        return metric.value.uiVal
    except Exception:
        return 0  # Log power as 0 if an exception occurs


@dataclass
class PowerReading:
    timestamp: float
    total_power: float


class NvmlPowerMonitor:

    def __init__(self,
                 interval: float,
                 csv_filename: str,
                 log_interval: float,
                 enable_mem_freq_meas: bool = False,
                 power_queue: Optional[multiprocessing.SimpleQueue] = None):
        self.interval = interval
        self.csv_filename = csv_filename
        self.log_interval = log_interval
        self.enable_mem_freq_meas = enable_mem_freq_meas
        self.logs: list[list[float]] = []
        self.stop_monitoring = False
        self.power_queue = power_queue

    def monitor_power_and_freq(self):
        pynvml.nvmlInit()
        try:
            cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is not None:
                visible_indices = [
                    int(x) for x in cuda_visible_devices.split(",")
                ]
            else:
                visible_indices = list(range(pynvml.nvmlDeviceGetCount()))
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in visible_indices
            ]

            column_names = ["Timestamp"]
            for i in range(len(handles)):
                column_names.append(f"GPU_{i}_power_w")
                column_names.append(f"GPU_{i}_freq_mhz")
                if self.enable_mem_freq_meas:
                    column_names.append(f"GPU_{i}_mem_freq_mhz")

            os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
            if os.path.exists(self.csv_filename):
                os.remove(self.csv_filename)
            with open(self.csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(column_names)

            logger.info('Monitoring power and frequency for %d GPUs...',
                        len(handles))
            last_log_time = time.perf_counter()

            while not self.stop_monitoring:
                timestamp = time.perf_counter()
                readings = [timestamp]
                total_power = 0
                for handle in handles:
                    power_usage = get_nvml_metric_value(
                        handle, pynvml.NVML_FI_DEV_POWER_INSTANT) / 1000.0
                    freq = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS)
                    readings.extend([power_usage, freq])
                    total_power += power_usage
                    if self.enable_mem_freq_meas:
                        mem_freq = pynvml.nvmlDeviceGetClockInfo(
                            handle, pynvml.NVML_CLOCK_MEM)
                        readings.append(mem_freq)

                self.logs.append(readings)

                if self.power_queue is not None:
                    self.power_queue.put(
                        PowerReading(timestamp=timestamp,
                                     total_power=total_power))

                if timestamp - last_log_time >= self.log_interval:
                    self._write_logs_to_csv()
                    last_log_time = timestamp

                time.sleep(self.interval)

            self._write_logs_to_csv()

        finally:
            pynvml.nvmlShutdown()

    def _write_logs_to_csv(self):
        if self.logs:
            with open(self.csv_filename, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(self.logs)
            logger.debug('Appended %d log entries to %s', len(self.logs),
                         self.csv_filename)
            self.logs = []


def start_nvml_power_monitor(
        interval: float,
        csv_filename: str,
        log_interval=1,
        enable_mem_freq_meas=False,
        power_queue: Optional[multiprocessing.SimpleQueue] = None):
    monitor = NvmlPowerMonitor(interval=interval,
                               csv_filename=csv_filename,
                               log_interval=log_interval,
                               enable_mem_freq_meas=enable_mem_freq_meas,
                               power_queue=power_queue)
    monitor.monitor_power_and_freq()


@contextlib.contextmanager
def measure_power(csv_filename,
                  interval=0.1,
                  log_interval=0.1,
                  enable_mem_freq_meas=False,
                  power_queue: Optional[multiprocessing.SimpleQueue] = None):
    process = multiprocessing.Process(target=start_nvml_power_monitor,
                                      args=(interval, csv_filename,
                                            log_interval, enable_mem_freq_meas,
                                            power_queue))
    process.start()
    try:
        logger.info("Power monitoring process starting ...")
        yield
    finally:
        process.terminate()
        process.join()
        logger.info("Power monitoring process terminated.")