# Disaggregated Serving Launch Scripts

This folder provides a config-driven launcher for disaggregated vLLM serving with:

- `n1` prefill instances and `n2` decode instances
- per-instance GPU assignment and TP degree
- optional per-iteration profiling (`--iter-profile`, `--iter-profile-dir`)
- per-instance max batch settings and chunk size
- automatic proxy startup after backend readiness
- separate aiPerf benchmark runner

## Files

- `launch_disagg_cluster.py`: Main cluster launcher.
- `launcher_lib.py`: Shared launch/config/readiness/utilities.
- `disagg_proxy_pool.py`: Round-robin proxy for prefill/decode backends.
- `run_aiperf_benchmark.py`: aiPerf runner.
- `stop_disagg_cluster.py`: Stops vLLM instances, proxy, and optional benchmark processes.
- `configs/disagg_cluster_config.2_gpus.json`: 2-GPU launch config (iter profiling enabled).
- `configs/disagg_cluster_config_8gpus.json`: 8-GPU launch config.

## Launch Cluster

```bash
python3 launch_disagg_cluster.py --config configs/disagg_cluster_config.2_gpus.json --dry-run
python3 launch_disagg_cluster.py --config configs/disagg_cluster_config.2_gpus.json
```

To skip proxy startup:

```bash
python3 launch_disagg_cluster.py --config configs/disagg_cluster_config.2_gpus.json --skip-proxy
```

## aiPerf Benchmark

```bash
python3 run_aiperf_benchmark.py --config configs/aiperf_benchmark_config.json
```

## Stop Cluster and Benchmark

```bash
python3 stop_disagg_cluster.py --config configs/disagg_cluster_config.2_gpus.json
```

Useful variants:

```bash
# Stop a specific run id under main_log_dir
python3 stop_disagg_cluster.py --config configs/disagg_cluster_config.2_gpus.json --run-id qwen_disagg_example_2gpu

# Stop using an explicit state file
python3 stop_disagg_cluster.py --state-file /export2/obasit/ClusterMoE/logs/qwen1.5_2.7B/qwen_disagg_example_2gpu/cluster_state.json

# Preview what would be stopped
python3 stop_disagg_cluster.py --config configs/disagg_cluster_config.2_gpus.json --dry-run
```

Stop behavior:

- Reads `cluster_state.json` (latest run by default when config is provided, or explicit run/state file).
- Stops prefill/decode vLLM instances and proxy via process group signals.
- Also stops benchmark processes (`aiperf profile`, `run_aiperf_benchmark.py`, `run_aiperf_benchmark.sh`) unless `--skip-benchmark` is set.
- Falls back to process-pattern matching if state-based PIDs are missing.

## Main Config Knobs

Top-level keys:

- `model`: Model name.
- `working_dir`: Where `vllm serve` and proxy process run.
- `main_log_dir`: Parent log directory.
- `gpu_pool`: Available GPUs for auto-assignment.
- `kv_transfer_backend`: KV transport backend (`lmcache` or `nixl`).
- `iter_profile.enabled`: Enables per-iteration profiling for all instances unless overridden.
- `iter_profile.subdir`: Subdirectory inside each instance log directory for iter-profile JSONL files.
- `prefill.count`: Number of prefill instances.
- `decode.count`: Number of decode instances.
- `prefill.tp_size`, `decode.tp_size`: Default TP degree per role.
- `prefill.gpu_groups`, `decode.gpu_groups`: Explicit GPU groups per instance.
- `prefill.instance_overrides`, `decode.instance_overrides`: Per-instance overrides.
- `proxy.host`, `proxy.port`: Public proxy endpoint.

Per-role and per-instance fields:

- `max_num_seqs` -> `--max-num-seqs`
- `max_num_batched_tokens` -> `--max-num-batched-tokens`
- `chunk_size` -> `LMCACHE_CHUNK_SIZE`
- `enable_prefix_caching` -> `--enable-prefix-caching` / `--no-enable-prefix-caching`
- `enable_expert_parallel` -> `--enable-expert-parallel`
- `enable_iter_profile` -> `--iter-profile`
- `iter_profile_log_subdir` -> controls directory passed via `--iter-profile-dir`
- `nixl_side_channel_port_start` / `nixl_side_channel_port` -> `VLLM_NIXL_SIDE_CHANNEL_PORT`
- `nixl_side_channel_host` -> `VLLM_NIXL_SIDE_CHANNEL_HOST`
- `kv_transfer_backend` -> optional role/instance override for connector choice
- `kv_connector_extra_config` -> merged into `kv_transfer_config.kv_connector_extra_config`
- `kv_transfer_fields` -> extra top-level KV fields (for example `kv_load_failure_policy`)
- `tp_size`
- `gpu_ids`
- `port`
- `extra_vllm_args`

## KV Backend Notes

- `kv_transfer_backend=lmcache` (default): launcher uses LMCache-style KV transfer config and sets `LMCACHE_CONFIG_FILE`; `chunk_size` applies here.
- `kv_transfer_backend=nixl`: launcher uses `NixlConnector` by default (unless overridden by `kv_connector`) and does not require LMCache config paths.
- For `kv_transfer_backend=nixl`, launcher sets a unique side-channel port per instance by default (`10000 + serve_port`) to avoid bind collisions.
- You can set backend globally and override per role/instance.

## Path Resolution

- `working_dir` and `main_log_dir` are resolved relative to the config file if not absolute.
- Relative `lmcache_config_file` paths are checked relative to the config file first, then `working_dir`.

## Logging Layout

For each run, launcher creates:

```text
<main_log_dir>/<run_id>/
	cluster_state.json
	prefill_i<idx>_p<port>_g<gpus>/
		vllm.log
		<iter_profile.subdir>/
			iter_profile_instance_<...>.jsonl
	decode_i<idx>_p<port>_g<gpus>/
		vllm.log
		<iter_profile.subdir>/
			iter_profile_instance_<...>.jsonl
	proxy/
		proxy.log
```

## Notes

- The launcher waits for each instance readiness endpoint before continuing.
- After all backends are ready, it launches and verifies proxy readiness.
- Processes run in background process groups.
- Deprecated profiling flags are no longer emitted by these scripts.
