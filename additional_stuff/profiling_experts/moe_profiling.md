# MoE Routing Profiling

This guide explains how to enable lightweight MoE routing profiling and inspect
the generated per-request logs.

## What `--enable-moe-profiling` writes

When enabled, vLLM writes per-request profiling output into
`--moe-profiling-log-dir` (default: `./vllm_moe_profiles`).

- Frontend per-request JSONL records with prompt/output text and tokens plus
  per-iteration, per-layer expert activations and full expert probability maps

Legacy `moe_profile_*.pt` chunk output is disabled.

Per-request log file format:

```text
moe_request_profile_pid<pid>_ts<YYYYMMDDTHHMMSSZ>_idx<index>_req<request_id>_ireq<internal_id>.msgpack.zlib
```

Each finished request is written to its own compressed binary file.
The logging path is asynchronous (background writer threads) so inference does
not block on serialization and disk I/O.

Each JSONL line is one finished request with:

- `request_id` and `internal_request_id`
- `input.text` and `input.token_ids`
- `output.text` and `output.token_ids`
- `moe_expert_activation`: list of iterations
  - `iter_no`, `token_count`
  - `layer0_input_embedding`: list shaped `[token_count, hidden_size]`
  - `layers`: list of `{layer_no, expert_ids, expert_probabilities}`
    - `expert_probabilities` is shaped `[token_count, num_experts]`
    - For each layer/token, taking top-k along the expert axis reproduces the
      selected `expert_ids` in standard top-k routing modes

## What `--enable-temporal-expert-logging` writes

When enabled, workers write temporal expert logs into
`--temporal-expert-log-dir` (default: `./vllm_temporal_expert_logs`).

- Iteration-level snapshots of top-k routed experts (k comes from model config)
- Grouped by request position within that iteration's active batch
- No request IDs or prompt/output payload are stored

Chunk file format:

```text
temporal_expert_profile_<instance>_dp<dp_rank>_<tp|ep><rank>_pid<pid>_ts<YYYYMMDDTHHMMSSZ>_chunk<index>.msgpack.zlib
```

When expert parallelism is enabled, files are keyed by `ep<rank>` so each
EP instance writes a distinct stream.

Each chunk contains:

- `iteration_no`
- `token_count`
- `request_token_counts` (token count for each request-position in batch order)
- `iteration_time_ms` (optional CUDA elapsed time for the full worker
  iteration on the worker stream, with no explicit stream synchronization)
- `layers`
  - `layer_no`
  - `request_expert_ids` shaped `[num_requests, tokens_for_request, <=top_k]`
    (global expert IDs are retained, and experts non-local to the current
    serving instance are dropped)

Writes are buffered and flushed periodically as compressed chunks using a
background writer thread to reduce impact on inference latency and memory.

## Enable MoE profiling

### Serve + benchmark example (`vllm bench serve`)

1. Start a vLLM server with profiling enabled:

```bash
vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --host 0.0.0.0 \
  --port 9000 \
  --enable-moe-profiling \
  --moe-profiling-log-dir ./moe_logs \
  --enable-temporal-expert-logging \
  --temporal-expert-log-dir ./temporal_expert_logs
```

2. In another terminal, run online serving benchmark traffic:

```bash
vllm bench serve --model Qwen/Qwen1.5-MoE-A2.7B-Chat --host 127.0.0.1 --port 9000 --request-rate 5 --random-input-len 128 --random-output-len 32 --num-prompts 10 --random-range-ratio 0.5
```

After benchmark completion, per-request profiling JSONL is available under
`./moe_logs`.

### Disaggregated serving example (GPU 0 prefill, GPU 1 decode)

This example uses the same MoE model as the live disaggregated run:
`Qwen/Qwen1.5-MoE-A2.7B-Chat`.

Prerequisites:

- Activate `vllm-moe` conda env
- Install `lmcache` and `nixl` in that env

From repo root (`vllm-MoE`), run in separate terminals.

1. Start prefill node on GPU 0 (writes profiling logs to `./moe_logs/prefill`):

```bash
cd examples/others/lmcache/disagg_prefill_lmcache_v1
source /home/obasit/miniconda3/etc/profile.d/conda.sh
conda activate vllm-moe

UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_CONFIG_FILE=./configs/lmcache-prefiller-config.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --host 0.0.0.0 \
  --port 8100 \
  --enforce-eager \
  --enable-moe-profiling \
  --moe-profiling-log-dir /export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/prefill \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"producer1"}}'
```

2. Start decode node on GPU 1 (writes profiling logs to `./moe_logs/decode`):

```bash
cd examples/others/lmcache/disagg_prefill_lmcache_v1
source /home/obasit/miniconda3/etc/profile.d/conda.sh
conda activate vllm-moe

UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_CONFIG_FILE=./configs/lmcache-decoder-config.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=1 \
vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --host 0.0.0.0 \
  --port 8200 \
  --enforce-eager \
  --enable-moe-profiling \
  --moe-profiling-log-dir /export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/decode \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"discard_partial_chunks":false,"lmcache_rpc_port":"consumer1"}}'
```

3. Start proxy server (single public endpoint on port 9000):

```bash
cd examples/others/lmcache/disagg_prefill_lmcache_v1
source /home/obasit/miniconda3/etc/profile.d/conda.sh
conda activate vllm-moe

python disagg_proxy_server.py \
  --host localhost \
  --port 9000 \
  --prefiller-host localhost \
  --prefiller-port 8100 \
  --decoder-host localhost \
  --decoder-port 8200
```

4. Send traffic through the proxy:

```bash
curl -sS http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen1.5-MoE-A2.7B-Chat","prompt":"Write one short sentence about MoE models.","max_tokens":24,"temperature":0.0}'
```

After requests complete, per-request profiling logs are in:

- `examples/others/lmcache/disagg_prefill_lmcache_v1/moe_logs/prefill`
- `examples/others/lmcache/disagg_prefill_lmcache_v1/moe_logs/decode`


# aiPerf running code

```bash
aiperf profile \
  --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --url http://localhost:9000 \
  --endpoint-type chat \
  --input-file /export2/obasit/ClusterMoE/vllm-MoE/additional_stuff/profiling_experts/mooncake_trace.jsonl \
  --custom-dataset-type mooncake_trace \
  --streaming \
  --concurrency 1 \
  --request-rate 1 \
  --arrival-pattern constant
```