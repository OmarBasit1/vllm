# SPDX-License-Identifier: Apache-2.0
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import uvloop

from benchmark_batch import BenchmarkBatchParam, benchmark_batch
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms.nvml_utils import get_gpu_name, get_preselected_freq
from vllm.utils import FlexibleArgumentParser

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

RESULT_ROOT = Path("/export2/obasit/MoE/logs")


def gen_from_trace(
    gpu: str,
    model: str,
    skip_existing: bool = True,
):
    log_dir_base = (
        RESULT_ROOT / f"request_timing/2025-05-05_lat-model-profiling/{gpu}_{model}"
    )

    # Batch shape traces are unchanged across GPU and models
    BATCH_SHAPE_TRACES: list[Path] = [
        Path(
            "/export2/kong102/energy_efficient_serving_results/request_timing/2025-05-05_batch-shape-profiling/A40_Llama-3.1-8B-Instruct_qps9_reqs20000_fixed1740/perf_metric_3321721.csv"
        ),
        Path(
            "/export2/kong102/energy_efficient_serving_results/request_timing/2025-05-05_batch-shape-profiling/A40_Llama-3.1-8B-Instruct_qps5_reqs20000_fixed1740/perf_metric_3378238.csv"
        ),
    ]

    max_counts: dict = {
        "prefill-only": 1000,
        "decode-only": 10000,
        "hybrid": 20000,
    }

    counter = Counter()
    params = []
    test_freqs = get_preselected_freq(gpu)

    for trace in BATCH_SHAPE_TRACES:
        df = pd.read_csv(trace)

        for idx, row in df.iterrows():
            num_computed_tokens = eval(row["num_precomputed_tokens_per_req_iter"])
            chunk_sizes = eval(row["chunk_size_per_req_iter"])

            # Skip over heartbeat rows
            if len(num_computed_tokens) == 0:
                continue

            prefill_lens = []
            prefill_computed_lens = []
            decode_lens = []

            start_decode_ind = 0

            for i, size in enumerate(chunk_sizes):
                if size > 1:
                    start_decode_ind = i + 1

            for i in range(start_decode_ind):
                prefill_lens.append(num_computed_tokens[i] + chunk_sizes[i])
                prefill_computed_lens.append(num_computed_tokens[i])

            for i in range(start_decode_ind, len(chunk_sizes)):
                decode_lens.append(num_computed_tokens[i] + chunk_sizes[i])

            for freq in test_freqs:
                tag = f"requests_{len(decode_lens) + len(prefill_lens)}_batch_{len(params):06d}_freq{freq}"  # noqa: E501
                p = BenchmarkBatchParam(
                    prefill_input_lens=prefill_lens,
                    prefill_completed_input_lens=prefill_computed_lens,
                    decode_input_lens=decode_lens,
                    log_dir=f"{log_dir_base}/logs/{tag}",
                    gpu_freq_mhz=freq,
                    min_num_iters=10,
                    min_seconds=1,
                )
                # Apply upper limit to each type
                batch_type = p.get_batch_type()
                if counter[batch_type] < max_counts[batch_type]:
                    params.append(p)
                    counter[batch_type] += 1

    # Supplement prefills by extracting from hybrid batches if needed
    # prefills_supp = []
    # for p in params:
    #     if p.get_batch_type() != 'hybrid':
    #         continue
    #     if counter['prefill-only'] >= max_counts['prefill-only']:
    #         break
    #     p_copy = copy.deepcopy(p)
    #     p_copy.decode_input_lens.clear()
    #     p_copy.log_dir = f"{log_dir_base}/logs/batch_{(len(params) + len(prefills_supp)):06d}_freq{p.gpu_freq_mhz}"  # noqa
    #     prefills_supp.append(p_copy)
    #     counter['prefill-only'] += 1
    # params.extend(prefills_supp)

    print("Batches per type: ", counter)

    if skip_existing:
        params = [p for p in params if not Path(p.log_dir).exists()]
    return params


def main(expr_fn: Callable, model: str):
    vllm_args = f"--model {model} -tp 1 -dp 2 --enable-chunked-prefill "

    # Keep it same with `benchmark_serving_driver.sh`
    gpu_name = get_gpu_name()
    if gpu_name == "A40" and model == "Qwen/Qwen1.5-MoE-A2.7B":
        vllm_args += (
            "--max-model-len 65536 --max-num-seqs 1024 --max-num-batched-tokens 1024 "
        )
    else:
        raise NotImplementedError(f"gpu: {gpu_name}, model: {model}")
    print("vllm_args: ", vllm_args)

    parser = FlexibleArgumentParser(description="Benchmark per-batch.")
    parser = AsyncEngineArgs.add_cli_args(parser)
    vllm_args = parser.parse_args(vllm_args.split())
    engine_args = AsyncEngineArgs.from_cli_args(vllm_args)

    # Pass in a list instead of generator so tqdm prints progress
    params = expr_fn(get_gpu_name(), model.split("/")[1])

    uvloop.run(benchmark_batch(engine_args, params))


if __name__ == "__main__":
    expr_fn = {
        # 'idle-power': gen_benchmark_idle_power_args,
        # 'sarathi-serve-sla': gen_sarathi_args,
        "trace": gen_from_trace,
    }[sys.argv[1]]
    model = sys.argv[2]
    main(expr_fn, model)
