# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import uvloop

# --- Core vLLM imports ---
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import nvml_get_available_freq, nvml_lock_freq
from vllm.sampling_params import SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import CSVLogger


# --- The new, detailed BenchmarkBatchParam class ---
@dataclass
class BenchmarkBatchParam:
    prefill_input_lens: list[int]
    decode_input_lens: list[int]
    log_dir: str
    gpu_freq_mhz: int
    prefill_completed_input_lens: list[int] = field(default_factory=list)

    # Delay before issuing each batch, drawn from uniform distribution
    delay_time_min_s: float = 0.0
    delay_time_max_s: float = 0.0

    # Run terminates when both are reached
    min_num_iters: int = 32
    min_seconds: int = 5

    def __post_init__(self):
        # Ensure completed lengths list matches prefill lengths list if provided
        if self.prefill_completed_input_lens and len(
            self.prefill_completed_input_lens
        ) != len(self.prefill_input_lens):
            raise ValueError(
                "prefill_completed_input_lens must have the same number of "
                "elements as prefill_input_lens"
            )

    def __hash__(self):
        hash_str = str(self.prefill_input_lens)
        hash_str += str(self.decode_input_lens)
        hash_str += str(self.gpu_freq_mhz)
        hash_str += str(self.delay_time_min_s)
        hash_str += str(self.delay_time_max_s)
        return int(hashlib.md5(hash_str.encode()).hexdigest(), 16)

    def get_batch_type(self) -> str:
        if len(self.prefill_input_lens) > 0 and len(self.decode_input_lens):
            return "hybrid"
        elif len(self.prefill_input_lens) > 0:
            return "prefill-only"
        elif len(self.decode_input_lens) > 0:
            return "decode-only"
        else:
            raise RuntimeError("Malformed BenchmarkBatchParam")

    def get_total_requests(self) -> int:
        return len(self.prefill_input_lens) + len(self.decode_input_lens)

    def get_batch_description(self) -> str:
        return (
            f"prefill_chunks={len(self.prefill_input_lens)}, "
            f"decode_seqs={len(self.decode_input_lens)}, "
            f"total_requests={self.get_total_requests()}"
        )


# --- Helper functions ---


async def consume_generator(generator):
    """Consumes an async generator to drive it to completion."""
    async for _ in generator:
        break


async def execute_benchmark_scenario(
    llm: AsyncLLM, tokenizer, param: BenchmarkBatchParam, engine_args: AsyncEngineArgs
):
    """
    Manages a single benchmark scenario.
    """
    # change logger path
    if llm.stat_loggers:
        for stat_loggers in llm.stat_loggers:
            for stat_logger in stat_loggers:
                if isinstance(stat_logger, CSVLogger):
                    stat_logger.filename = (
                        Path(param.log_dir) / f"engine_{stat_logger.engine_index}.csv"
                    )
                    stat_logger.persist_to_disk_every = 10

                    stat_logger.filename.parent.mkdir(parents=True, exist_ok=True)
                    if stat_logger.filename.exists():
                        stat_logger.filename.unlink()

    request_definitions = []
    # A. Define PREFILL requests
    for i, total_len in enumerate(param.prefill_input_lens):
        completed_len = (
            param.prefill_completed_input_lens[i]
            if param.prefill_completed_input_lens
            else 0
        )
        prompt_len = total_len - completed_len
        if prompt_len > 0:
            request_definitions.append({"type": "prefill", "prompt_len": prompt_len})

    # B. Define DECODE requests
    for i, input_len in enumerate(param.decode_input_lens):
        prompt_len = input_len - 1
        if prompt_len > 0:
            request_definitions.append({"type": "decode", "prompt_len": prompt_len})

    if not request_definitions:
        print("No requests to process in this scenario. Skipping.")
        return

    print(f"Executing Scenario: {param.get_batch_description()}")
    print(f"Locking GPU frequency to {param.gpu_freq_mhz} MHz...")

    with nvml_lock_freq(param.gpu_freq_mhz):
        start_time = time.perf_counter()
        iter_num = 0
        while True:
            iter_num += 1
            # Check for time-based termination first
            elapsed_time = time.perf_counter() - start_time
            if iter_num > param.min_num_iters and elapsed_time > param.min_seconds:
                break
            iter_tasks = []
            # ---------------- WARM-UP REQUEST ----------------
            # For some reason in TP, the first request is batched separately
            # and the rest are batched together. So warmup is run separately while
            # and the others are batched together. Remove this in post processing.
            if engine_args.tensor_parallel_size > 1:
                warmup_prompt = "Warm"
                warmup_params = SamplingParams(max_tokens=1, ignore_eos=True)
                warmup_gen = llm.generate(
                    warmup_prompt,
                    warmup_params,
                    request_id=f"warmup-{iter_num}-{time.time()}",
                )
                iter_tasks.append(asyncio.create_task(consume_generator(warmup_gen)))
            # --------------------------------------------------

            # --- MAIN BATCH CREATION & EXECUTION ---

            for i, req_def in enumerate(request_definitions):
                random.seed(0)
                prompt_tokens = [
                    random.randint(0, tokenizer.vocab_size - 1)
                    for _ in range(req_def["prompt_len"])
                ]
                prompt = tokenizer.decode(prompt_tokens)
                sampling_params = SamplingParams(
                    n=1,
                    temperature=1.0,
                    top_p=1.0,
                    ignore_eos=True,
                    max_tokens=1,
                )
                request_id = f"{req_def['type']}-{i}-{time.time()}"
                gen = llm.generate(prompt, sampling_params, request_id=request_id)
                iter_tasks.append(asyncio.create_task(consume_generator(gen)))

            # This sends all requests to the engine at once.
            await asyncio.gather(*iter_tasks)

            # --- END OF MAIN BATCH ---

            # Apply delay between iterations
            if param.delay_time_max_s > 0:
                delay = random.uniform(param.delay_time_min_s, param.delay_time_max_s)
                await asyncio.sleep(delay)

        if llm.stat_loggers:
            for stat_loggers in llm.stat_loggers:
                for stat_logger in stat_loggers:
                    if isinstance(stat_logger, CSVLogger):
                        stat_logger.persist_to_disk()
    print("-" * 50)


async def benchmark_batch(
    engine_args: AsyncEngineArgs, benchmark_params: Iterable[BenchmarkBatchParam]
):
    """
    Main async function that initializes the engine once and then runs
    all benchmark scenarios.
    """
    print("Initializing vLLM engine...")
    llm = AsyncLLM.from_engine_args(engine_args)
    tokenizer = await llm.get_tokenizer()

    print("Engine is ready.")
    print("=" * 50)

    for param in benchmark_params:
        await execute_benchmark_scenario(llm, tokenizer, param, engine_args)


# --- Main entrypoint structured as you requested ---
if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark Data Parallelism with structured batches for vLLM v1."
    )
    # CRITICAL: We use AsyncEngineArgs because this script requires the AsyncLLM
    # engine to correctly test data parallelism via concurrent requests.
    parser = AsyncEngineArgs.add_cli_args(parser)

    # Example arguments provided as a string for easy testing.
    # You can also pass these arguments via the command line.
    vllm_args_str = (
        "--model Qwen/Qwen1.5-MoE-A2.7B "
        "--tensor-parallel-size 1 "
        "--data-parallel-size 2 "
        "--max-num-seqs 1024 "
        "--enable-chunked-prefill "
        "--max-num-batched-tokens 1024 "
    )

    parsed_args = parser.parse_args(vllm_args_str.split())
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)

    # Define the list of benchmark scenarios to run
    benchmark_params_list = [
        BenchmarkBatchParam(
            prefill_completed_input_lens=[],
            prefill_input_lens=[4] * 2,
            decode_input_lens=[],
            log_dir="logs/hybrid11",
            gpu_freq_mhz=nvml_get_available_freq()[-1],
            delay_time_min_s=0,
            delay_time_max_s=0,
            min_num_iters=5,
            min_seconds=0,
        ),
        BenchmarkBatchParam(
            prefill_completed_input_lens=[128, 64, 8],
            prefill_input_lens=[256] * 3,
            decode_input_lens=[512] * 4,
            log_dir="logs/hybrid33",
            gpu_freq_mhz=nvml_get_available_freq()[-1],
            delay_time_min_s=0,
            delay_time_max_s=0,
            min_num_iters=5,
            min_seconds=0,
        ),
    ]

    # Now, pass the correct 'engine_args' object to the function
    uvloop.run(benchmark_batch(engine_args, benchmark_params_list))
