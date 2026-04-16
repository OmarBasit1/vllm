# Disaggregated Serving Launch Scripts

This folder contains a modular, config-driven launcher for disaggregated vLLM serving with:

- `n1` prefill instances and `n2` decode instances
- per-instance GPU assignment and TP degree
- optional MoE profiling with per-instance log directories
- optional temporal expert logging with per-instance log directories
- per-instance max batch settings and chunk size
- automatic proxy startup after backend readiness
- separate aiPerf benchmark runner

## Files

- `launch_disagg_cluster.py`: Main cluster launcher.
- `launcher_lib.py`: Shared launch/config/readiness utilities.
- `disagg_proxy_pool.py`: Round-robin proxy for multiple prefill/decode backends.
- `run_aiperf_benchmark.py`: aiPerf runner.
- `stop_disagg_cluster.py`: Stopper for vLLM instances, proxy, and benchmark processes.
- `disagg_cluster_config.example.json`: Example launch config.
- `aiperf_benchmark_config.example.json`: Example aiPerf config.
- `launch_disagg_cluster.sh`: Shell wrapper for launcher.
- `run_aiperf_benchmark.sh`: Shell wrapper for aiPerf runner.
- `stop_disagg_cluster.sh`: Shell wrapper for stopper.

## Launch Cluster

Use one command with a JSON config:

```bash
cd /export2/obasit/ClusterMoE/vllm-MoE/additional_stuff/nebius_disag_launch_scripts
bash launch_disagg_cluster.sh disagg_cluster_config.example.json
```

Optional:

```bash
python3 launch_disagg_cluster.py --config disagg_cluster_config.example.json --dry-run
python3 launch_disagg_cluster.py --config disagg_cluster_config.example.json --skip-proxy
```

## aiPerf Benchmark

```bash
cd /export2/obasit/ClusterMoE/vllm-MoE/additional_stuff/nebius_disag_launch_scripts
bash run_aiperf_benchmark.sh aiperf_benchmark_config.example.json
```

## Stop Cluster and Benchmark

```bash
cd /export2/obasit/ClusterMoE/vllm-MoE/additional_stuff/nebius_disag_launch_scripts
bash stop_disagg_cluster.sh --config disagg_cluster_config.example.json
```

Useful variants:

```bash
# Stop a specific run id under main_log_dir
python3 stop_disagg_cluster.py --config disagg_cluster_config.example.json --run-id qwen_disagg_example_2gpu

# Stop using an explicit state file
python3 stop_disagg_cluster.py --state-file /export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/qwen_disagg_example_2gpu/cluster_state.json

# Preview what would be stopped
python3 stop_disagg_cluster.py --config disagg_cluster_config.example.json --dry-run
```

Stop behavior:

- Reads `cluster_state.json` (latest run by default when config is provided, or explicit run/state file).
- Stops prefill/decode vLLM instances and proxy via process group signals.
- Also stops benchmark processes (`aiperf profile`, `run_aiperf_benchmark.py`, `run_aiperf_benchmark.sh`) unless `--skip-benchmark` is set.
- Falls back to process-pattern matching for safety if state-based PIDs are missing.

## Main Config Knobs

Top-level keys in `disagg_cluster_config.example.json`:

- `model`: Model name.
- `working_dir`: Where `vllm serve` and proxy process run.
- `main_log_dir`: Parent log directory.
- `gpu_pool`: Available GPUs for auto-assignment.
- `kv_transfer_backend`: KV transport backend selector (`lmcache` or `nixl`).
- `prefill.count`: Number of prefill instances (`n1`).
- `decode.count`: Number of decode instances (`n2`).
- `prefill.tp_size`, `decode.tp_size`: Default TP degree per role.
- `prefill.enable_prefix_caching`, `decode.enable_prefix_caching`: Prefix-caching toggle per role (default `false`).
- `prefill.enable_expert_parallel`, `decode.enable_expert_parallel`: Enable vLLM expert parallel mode for that role (default `false`).
- `prefill.gpu_groups`, `decode.gpu_groups`: Explicit GPU groups per instance.
- `prefill.instance_overrides`, `decode.instance_overrides`: Per-instance overrides.
- `moe_profiling.enabled`: Global MoE profiling toggle.
- `temporal_expert_logging.enabled`: Global temporal expert logging toggle.
- `proxy.host`, `proxy.port`: Public proxy endpoint.

Per role and per-instance fields:

- `max_num_seqs` -> maps to `--max-num-seqs`
- `max_num_batched_tokens` -> maps to `--max-num-batched-tokens`
- `chunk_size` -> maps to `LMCACHE_CHUNK_SIZE`
- `enable_prefix_caching` -> maps to `--enable-prefix-caching` / `--no-enable-prefix-caching`
- `enable_expert_parallel` -> maps to `--enable-expert-parallel`
- `kv_transfer_backend` -> optional role/instance override for connector choice
- `kv_connector_extra_config` -> merged into `kv_transfer_config.kv_connector_extra_config`
- `kv_transfer_fields` -> extra top-level KV fields (e.g. `kv_load_failure_policy`)
- `tp_size`
- `gpu_ids`
- `port`
- `enable_moe_profiling`
- `enable_temporal_expert_logging`
- `temporal_expert_log_subdir`
- `extra_vllm_args`

KV backend notes:

- `kv_transfer_backend=lmcache` (default): launcher uses LMCache-style kv transfer config and sets `LMCACHE_CONFIG_FILE`; `chunk_size` applies here.
- `kv_transfer_backend=nixl`: launcher uses `NixlConnector` by default (unless you override `kv_connector`), and does not require LMCache config paths.
- You can switch backend globally at top-level or override per role / per instance.
- If you set `kv_connector` explicitly in role/instance config, that explicit connector overrides backend defaults.

Example NIXL knobs you can set under role or instance:

- `kv_connector_extra_config`: `{"backends": ["UCX", "GDS"]}`
- `kv_transfer_fields`: `{"kv_load_failure_policy": "fail", "kv_buffer_device": "cuda"}`

Path resolution notes:

- `working_dir` and `main_log_dir` are resolved relative to the config file if not absolute.
- Relative `lmcache_config_file` paths are first checked relative to the config file, then relative to `working_dir`.

## Logging Layout

For each run, launcher creates:

- `<main_log_dir>/<run_id>/cluster_state.json`
- `<main_log_dir>/<run_id>/prefill_i*_.../vllm.log`
- `<main_log_dir>/<run_id>/decode_i*_.../vllm.log`
- `<main_log_dir>/<run_id>/proxy/proxy.log`
- MoE logs under each instance dir if enabled.
- Temporal expert logs under each instance dir if enabled.

Each instance has its own identifiable directory under the run directory.

## Notes

- The launcher waits until each vLLM instance is ready before marking success.
- After all backends are ready, it launches and verifies the proxy server.
- Processes are launched in background process groups.
- Ensure your active environment has `vllm`, `lmcache`, `nixl`, `fastapi`, `httpx`, and `uvicorn`.
