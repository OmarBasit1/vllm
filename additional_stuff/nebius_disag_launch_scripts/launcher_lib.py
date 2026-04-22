#!/usr/bin/env python3
"""Shared utilities for launching disaggregated prefill/decode vLLM clusters."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_GLOBAL_ENV = {
    "UCX_TLS": "cuda_ipc,cuda_copy,tcp",
    "LMCACHE_USE_EXPERIMENTAL": "True",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "PYTHONHASHSEED": "123",
}


@dataclass
class InstanceRuntime:
    role: str
    index: int
    bind_host: str
    connect_host: str
    port: int
    tp_size: int
    gpu_ids: List[int]
    log_dir: Path
    log_file: Path
    iter_profile_log_dir: Optional[Path]
    command: List[str]
    env: Dict[str, str]
    process: subprocess.Popen[str]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def endpoint_ready(host: str, port: int, timeout_sec: float = 2.0) -> bool:
    candidate_paths = ["/health", "/v1/models", "/v1/completions"]
    for path in candidate_paths:
        url = f"http://{host}:{port}{path}"
        req = Request(url=url, method="GET")
        try:
            with urlopen(req, timeout=timeout_sec) as resp:
                if 200 <= resp.status < 500:
                    return True
        except HTTPError as exc:
            if exc.code in (401, 403, 405):
                return True
            if exc.code in (404,):
                continue
        except URLError:
            continue
        except TimeoutError:
            continue
    return False


def wait_for_ready_instances(
    instances: List[InstanceRuntime],
    timeout_sec: float,
    poll_interval_sec: float,
) -> None:
    start_times = {id(inst): time.time() for inst in instances}
    pending = {id(inst): inst for inst in instances}

    while pending:
        now = time.time()
        resolved: List[int] = []
        for key, inst in pending.items():
            if endpoint_ready(inst.connect_host, inst.port):
                print(
                    f"[ready] {inst.role}[{inst.index}] on {inst.connect_host}:{inst.port} "
                    f"(pid={inst.process.pid}, gpus={inst.gpu_ids}, tp={inst.tp_size})"
                )
                resolved.append(key)
                continue

            if inst.process.poll() is not None:
                tail = read_log_tail(inst.log_file, max_lines=40)
                raise RuntimeError(
                    "Process exited before readiness: "
                    f"{inst.role}[{inst.index}] pid={inst.process.pid}\n"
                    f"Log tail ({inst.log_file}):\n{tail}"
                )

            if now - start_times[key] > timeout_sec:
                tail = read_log_tail(inst.log_file, max_lines=40)
                raise TimeoutError(
                    "Timed out waiting for readiness: "
                    f"{inst.role}[{inst.index}] on {inst.connect_host}:{inst.port}\n"
                    f"Log tail ({inst.log_file}):\n{tail}"
                )

        for key in resolved:
            pending.pop(key, None)

        if pending:
            time.sleep(poll_interval_sec)


def read_log_tail(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return "<log file not found>"
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def _normalize_gpu_groups(
    role_name: str,
    role_cfg: Dict[str, Any],
    gpu_pool: List[int],
    gpu_cursor: int,
) -> Tuple[List[List[int]], int]:
    count = int(role_cfg["count"])
    default_tp = int(role_cfg.get("tp_size", 1))
    if default_tp < 1:
        raise ValueError(f"{role_name}.tp_size must be >= 1")

    custom_groups = role_cfg.get("gpu_groups")
    if custom_groups is not None:
        if len(custom_groups) != count:
            raise ValueError(
                f"{role_name}.gpu_groups must have length {count}, got {len(custom_groups)}"
            )
        normalized = []
        for idx, group in enumerate(custom_groups):
            if not isinstance(group, list) or not group:
                raise ValueError(
                    f"{role_name}.gpu_groups[{idx}] must be a non-empty list"
                )
            normalized.append([int(x) for x in group])
        return normalized, gpu_cursor

    needed = count * default_tp
    available = len(gpu_pool) - gpu_cursor
    if available < needed:
        raise ValueError(
            f"Insufficient GPUs for {role_name}: need {needed} from gpu_pool, "
            f"only {available} available"
        )

    groups: List[List[int]] = []
    for _ in range(count):
        groups.append(gpu_pool[gpu_cursor : gpu_cursor + default_tp])
        gpu_cursor += default_tp
    return groups, gpu_cursor


def _override_map(role_cfg: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    entries = role_cfg.get("instance_overrides", [])
    by_index: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        index = int(entry["index"])
        by_index[index] = entry
    return by_index


def _resolve_kv_transfer_backend(
    *,
    cfg: Dict[str, Any],
    role_cfg: Dict[str, Any],
    override: Dict[str, Any],
    role_name: str,
    index: int,
) -> str:
    backend_raw = override.get("kv_transfer_backend")
    if backend_raw is None:
        backend_raw = role_cfg.get("kv_transfer_backend")
    if backend_raw is None:
        backend_raw = cfg.get("kv_transfer_backend", "lmcache")

    backend = str(backend_raw).strip().lower()
    if backend not in ("lmcache", "nixl"):
        raise ValueError(
            f"{role_name}[{index}] kv_transfer_backend must be one of "
            "['lmcache', 'nixl']"
        )
    return backend


def _build_kv_transfer_config(
    *,
    backend: str,
    role_name: str,
    role_cfg: Dict[str, Any],
    override: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    if backend == "lmcache":
        kv_connector_default = "LMCacheConnectorV1"
        kv_extra: Dict[str, Any] = {
            "discard_partial_chunks": bool(
                override.get(
                    "discard_partial_chunks",
                    role_cfg.get("discard_partial_chunks", False),
                )
            ),
            "lmcache_rpc_port": str(
                override.get(
                    "rpc_port",
                    f"{role_cfg.get('rpc_port_prefix', role_name)}{index + 1}",
                )
            ),
        }
    else:
        kv_connector_default = "NixlConnector"
        kv_extra = {}

    kv_extra.update(role_cfg.get("kv_connector_extra_config", {}))
    kv_extra.update(override.get("kv_connector_extra_config", {}))

    kv_connector_raw = override.get("kv_connector")
    if kv_connector_raw is None:
        kv_connector_raw = role_cfg.get("kv_connector")
    if kv_connector_raw is None:
        kv_connector_raw = kv_connector_default

    kv_role_raw = override.get("kv_role")
    if kv_role_raw is None:
        kv_role_raw = role_cfg.get("kv_role")
    if kv_role_raw is None:
        kv_role_raw = "kv_both"

    kv_transfer_cfg: Dict[str, Any] = {
        "kv_connector": str(kv_connector_raw),
        "kv_role": str(kv_role_raw),
    }
    if kv_extra:
        kv_transfer_cfg["kv_connector_extra_config"] = kv_extra

    kv_transfer_fields = dict(role_cfg.get("kv_transfer_fields", {}))
    kv_transfer_fields.update(override.get("kv_transfer_fields", {}))

    for reserved in ("kv_connector", "kv_role", "kv_connector_extra_config"):
        if reserved in kv_transfer_fields:
            raise ValueError(
                f"Do not set '{reserved}' in kv_transfer_fields; set it directly in the role/override config"
            )

    kv_transfer_cfg.update(kv_transfer_fields)
    return kv_transfer_cfg


def _build_instance_cmd(
    *,
    vllm_command: str,
    model: str,
    role_cfg: Dict[str, Any],
    port: int,
    tp_size: int,
    max_num_seqs: Optional[int],
    max_num_batched_tokens: Optional[int],
    enable_prefix_caching: bool,
    enable_expert_parallel: bool,
    kv_transfer_cfg: Dict[str, Any],
    enable_iter_profile: bool,
    iter_profile_log_dir: Optional[Path],
    extra_common_args: List[str],
    extra_role_args: List[str],
    extra_instance_args: List[str],
) -> List[str]:
    cmd: List[str] = [
        vllm_command,
        "serve",
        model,
        "--host",
        str(role_cfg.get("bind_host", "0.0.0.0")),
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp_size),
    ]

    if bool(role_cfg.get("enforce_eager", True)):
        cmd.append("--enforce-eager")
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")

    if enable_expert_parallel:
        cmd.append("--enable-expert-parallel")

    if enable_iter_profile:
        if iter_profile_log_dir is None:
            raise ValueError(
                "iter_profile_log_dir must be set if iter_profile is enabled"
            )
        cmd.extend(
            [
                "--iter-profile",
                "--iter-profile-dir",
                str(iter_profile_log_dir),
            ]
        )

    cmd.extend(["--kv-transfer-config", json.dumps(kv_transfer_cfg, separators=(",", ":"))])

    cmd.extend(extra_common_args)
    cmd.extend(extra_role_args)
    cmd.extend(extra_instance_args)

    return cmd


def build_instances_and_commands(
    raw_cfg: Dict[str, Any],
    *,
    config_path: Path,
    run_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cfg = _deep_merge_dict(
        {
            "vllm_command": "vllm",
            "python_executable": "python3",
            "kv_transfer_backend": "lmcache",
            "launcher": {
                "health_check_timeout_sec": 900,
                "health_check_interval_sec": 2,
            },
            "proxy": {
                "host": "127.0.0.1",
                "port": 9000,
                "request_timeout_sec": 600,
            },
            "iter_profile": {
                "enabled": False,
                "subdir": "iter_profiles",
            },
            "global_env": {},
            "common_vllm_args": [],
            "gpu_pool": [],
            "working_dir": ".",
            "prefill": {
                "count": 1,
                "tp_size": 1,
                "port_start": 8100,
                "port_step": 1,
                "connect_host": "127.0.0.1",
                "bind_host": "0.0.0.0",
                "nixl_side_channel_host": None,
                "nixl_side_channel_port_start": None,
                "kv_transfer_backend": None,
                "kv_connector": None,
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {},
                "kv_transfer_fields": {},
                "rpc_port_prefix": "producer",
                "discard_partial_chunks": False,
                "lmcache_config_file": "./configs/lmcache-prefiller-config.yaml",
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "enable_expert_parallel": False,
                "enable_iter_profile": None,
                "iter_profile_log_subdir": None,
                "max_num_seqs": None,
                "max_num_batched_tokens": None,
                "chunk_size": None,
                "extra_vllm_args": [],
                "instance_overrides": [],
            },
            "decode": {
                "count": 1,
                "tp_size": 1,
                "port_start": 8200,
                "port_step": 1,
                "connect_host": "127.0.0.1",
                "bind_host": "0.0.0.0",
                "nixl_side_channel_host": None,
                "nixl_side_channel_port_start": None,
                "kv_transfer_backend": None,
                "kv_connector": None,
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {},
                "kv_transfer_fields": {},
                "rpc_port_prefix": "consumer",
                "discard_partial_chunks": False,
                "lmcache_config_file": "./configs/lmcache-decoder-config.yaml",
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "enable_expert_parallel": False,
                "enable_iter_profile": None,
                "iter_profile_log_subdir": None,
                "max_num_seqs": None,
                "max_num_batched_tokens": None,
                "chunk_size": None,
                "extra_vllm_args": [],
                "instance_overrides": [],
            },
        },
        raw_cfg,
    )

    if "model" not in cfg or not cfg["model"]:
        raise ValueError("config must include a non-empty 'model'")

    base_dir = config_path.parent
    cfg["working_dir"] = str(resolve_path(cfg["working_dir"], base_dir))
    working_dir_path = Path(cfg["working_dir"])
    if not working_dir_path.is_dir():
        raise ValueError(f"working_dir does not exist or is not a directory: {working_dir_path}")

    main_log_dir = cfg.get("main_log_dir")
    if not main_log_dir:
        raise ValueError("config must include 'main_log_dir'")
    cfg["main_log_dir"] = str(resolve_path(main_log_dir, base_dir))

    if not cfg["gpu_pool"]:
        raise ValueError("config must include non-empty gpu_pool list")
    gpu_pool = [int(g) for g in cfg["gpu_pool"]]

    global_env = dict(DEFAULT_GLOBAL_ENV)
    global_env.update({k: str(v) for k, v in cfg.get("global_env", {}).items()})

    all_instances: List[Dict[str, Any]] = []
    gpu_cursor = 0

    for role_name in ("prefill", "decode"):
        role_cfg = cfg[role_name]
        count = int(role_cfg["count"])
        if count < 1:
            raise ValueError(f"{role_name}.count must be >= 1")

        gpu_groups, gpu_cursor = _normalize_gpu_groups(
            role_name=role_name,
            role_cfg=role_cfg,
            gpu_pool=gpu_pool,
            gpu_cursor=gpu_cursor,
        )

        overrides = _override_map(role_cfg)
        port_start = int(role_cfg.get("port_start", 0))
        port_step = int(role_cfg.get("port_step", 1))
        if port_start <= 0 or port_step <= 0:
            raise ValueError(f"{role_name}.port_start and port_step must be positive")

        for idx in range(count):
            override = overrides.get(idx, {})
            assigned_gpus = [int(x) for x in override.get("gpu_ids", gpu_groups[idx])]
            tp_size = int(override.get("tp_size", role_cfg.get("tp_size", len(assigned_gpus))))
            if tp_size < 1:
                raise ValueError(f"{role_name}[{idx}] tp_size must be >= 1")
            if len(assigned_gpus) != tp_size:
                raise ValueError(
                    f"{role_name}[{idx}] gpu_ids count ({len(assigned_gpus)}) "
                    f"must match tp_size ({tp_size})"
                )

            port = int(override.get("port", port_start + idx * port_step))

            max_num_seqs_raw = override.get("max_num_seqs", role_cfg.get("max_num_seqs"))
            max_num_batched_tokens_raw = override.get(
                "max_num_batched_tokens", role_cfg.get("max_num_batched_tokens")
            )
            max_num_seqs = (
                int(max_num_seqs_raw) if max_num_seqs_raw is not None else None
            )
            max_num_batched_tokens = (
                int(max_num_batched_tokens_raw)
                if max_num_batched_tokens_raw is not None
                else None
            )

            enable_prefix_caching = bool(
                override.get(
                    "enable_prefix_caching",
                    role_cfg.get("enable_prefix_caching", False),
                )
            )
            enable_expert_parallel = bool(
                override.get(
                    "enable_expert_parallel",
                    role_cfg.get("enable_expert_parallel", False),
                )
            )

            chunk_size_raw = override.get("chunk_size", role_cfg.get("chunk_size"))
            chunk_size = int(chunk_size_raw) if chunk_size_raw is not None else None

            kv_transfer_backend = _resolve_kv_transfer_backend(
                cfg=cfg,
                role_cfg=role_cfg,
                override=override,
                role_name=role_name,
                index=idx,
            )

            kv_transfer_cfg = _build_kv_transfer_config(
                backend=kv_transfer_backend,
                role_name=role_name,
                role_cfg=role_cfg,
                override=override,
                index=idx,
            )

            instance_name = (
                f"{role_name}_i{idx}_p{port}_g{'-'.join(str(g) for g in assigned_gpus)}"
            )
            instance_log_dir = run_dir / instance_name
            instance_log_dir.mkdir(parents=True, exist_ok=True)

            enable_iter_profile_raw = override.get("enable_iter_profile")
            if enable_iter_profile_raw is None:
                enable_iter_profile_raw = role_cfg.get("enable_iter_profile")
            if enable_iter_profile_raw is None:
                enable_iter_profile_raw = cfg.get("iter_profile", {}).get(
                    "enabled", False
                )
            enable_iter_profile = bool(enable_iter_profile_raw)

            iter_profile_log_dir: Optional[Path] = None
            if enable_iter_profile:
                iter_profile_subdir_raw = override.get("iter_profile_log_subdir")
                if iter_profile_subdir_raw is None:
                    iter_profile_subdir_raw = role_cfg.get("iter_profile_log_subdir")
                if iter_profile_subdir_raw is None:
                    iter_profile_subdir_raw = cfg.get("iter_profile", {}).get(
                        "subdir", "iter_profiles"
                    )

                iter_profile_subdir = str(iter_profile_subdir_raw)
                iter_profile_log_dir = instance_log_dir / iter_profile_subdir
                iter_profile_log_dir.mkdir(parents=True, exist_ok=True)

            env = dict(os.environ)
            env.update(global_env)
            
            # --- Dynamically inject CONDA_PREFIX/lib into LD_LIBRARY_PATH if it's missing ---
            conda_prefix = env.get("CONDA_PREFIX")
            if conda_prefix:
                conda_lib_path = os.path.join(conda_prefix, "lib")
                current_ld_lib_path = env.get("LD_LIBRARY_PATH", "")
                
                # Check if it's already there to prevent duplicates
                if conda_lib_path not in current_ld_lib_path.split(os.pathsep):
                    if current_ld_lib_path:
                        env["LD_LIBRARY_PATH"] = f"{conda_lib_path}{os.pathsep}{current_ld_lib_path}"
                    else:
                        env["LD_LIBRARY_PATH"] = conda_lib_path
            # ----------------------------------------------------------------------------------

            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in assigned_gpus)
            if kv_transfer_backend == "lmcache":
                lmcache_cfg_text = str(
                    override.get("lmcache_config_file", role_cfg.get("lmcache_config_file"))
                )
                lmcache_cfg_path = Path(lmcache_cfg_text).expanduser()
                if lmcache_cfg_path.is_absolute():
                    lmcache_config_file = lmcache_cfg_path
                else:
                    from_config_dir = (base_dir / lmcache_cfg_path).resolve()
                    from_working_dir = (working_dir_path / lmcache_cfg_path).resolve()
                    if from_config_dir.exists():
                        lmcache_config_file = from_config_dir
                    elif from_working_dir.exists():
                        lmcache_config_file = from_working_dir
                    else:
                        # Prefer working_dir semantics for unresolved relative paths.
                        lmcache_config_file = from_working_dir

                env["LMCACHE_CONFIG_FILE"] = str(lmcache_config_file)
                if chunk_size is not None:
                    env["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
            else:
                nixl_side_channel_host = override.get(
                    "nixl_side_channel_host",
                    role_cfg.get("nixl_side_channel_host"),
                )
                if nixl_side_channel_host is not None:
                    env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = str(nixl_side_channel_host)

                nixl_side_channel_port_raw = override.get("nixl_side_channel_port")
                if nixl_side_channel_port_raw is None:
                    port_start_raw = role_cfg.get("nixl_side_channel_port_start")
                    if port_start_raw is not None:
                        nixl_side_channel_port_raw = int(port_start_raw) + idx

                # Default to a deterministic per-instance port derived from serve port.
                if nixl_side_channel_port_raw is None:
                    nixl_side_channel_port_raw = 10000 + port

                nixl_side_channel_port = int(nixl_side_channel_port_raw)
                if nixl_side_channel_port <= 0 or nixl_side_channel_port > 65535:
                    raise ValueError(
                        f"{role_name}[{idx}] nixl_side_channel_port must be in range 1..65535, "
                        f"got {nixl_side_channel_port}"
                    )
                env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(nixl_side_channel_port)

            cmd = _build_instance_cmd(
                vllm_command=str(cfg.get("vllm_command", "vllm")),
                model=str(cfg["model"]),
                role_cfg=role_cfg,
                port=port,
                tp_size=tp_size,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_prefix_caching=enable_prefix_caching,
                enable_expert_parallel=enable_expert_parallel,
                kv_transfer_cfg=kv_transfer_cfg,
                enable_iter_profile=enable_iter_profile,
                iter_profile_log_dir=iter_profile_log_dir,
                extra_common_args=[str(x) for x in cfg.get("common_vllm_args", [])],
                extra_role_args=[str(x) for x in role_cfg.get("extra_vllm_args", [])],
                extra_instance_args=[str(x) for x in override.get("extra_vllm_args", [])],
            )
            print(f"Built command for {role_name}[{idx}]: {' '.join(cmd)}")

            all_instances.append(
                {
                    "role": role_name,
                    "index": idx,
                    "bind_host": str(role_cfg.get("bind_host", "0.0.0.0")),
                    "connect_host": str(role_cfg.get("connect_host", "127.0.0.1")),
                    "port": port,
                    "tp_size": tp_size,
                    "gpu_ids": assigned_gpus,
                    "kv_transfer_backend": kv_transfer_backend,
                    "log_dir": str(instance_log_dir),
                    "log_file": str(instance_log_dir / "vllm.log"),
                    "iter_profile_log_dir": (
                        str(iter_profile_log_dir)
                        if iter_profile_log_dir is not None
                        else None
                    ),
                    "command": cmd,
                    "env": env,
                }
            )

    return cfg, all_instances


def launch_instance(
    instance_cfg: Dict[str, Any],
    *,
    working_dir: Path,
) -> InstanceRuntime:
    log_file = Path(instance_cfg["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_file.open("a", encoding="utf-8")
    process = subprocess.Popen(
        instance_cfg["command"],
        cwd=str(working_dir),
        env=instance_cfg["env"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    # Keep FD alive through child lifetime by keeping object attached to process.
    process._log_handle = log_handle  # type: ignore[attr-defined]

    return InstanceRuntime(
        role=str(instance_cfg["role"]),
        index=int(instance_cfg["index"]),
        bind_host=str(instance_cfg["bind_host"]),
        connect_host=str(instance_cfg["connect_host"]),
        port=int(instance_cfg["port"]),
        tp_size=int(instance_cfg["tp_size"]),
        gpu_ids=[int(x) for x in instance_cfg["gpu_ids"]],
        log_dir=Path(instance_cfg["log_dir"]),
        log_file=log_file,
        iter_profile_log_dir=Path(instance_cfg["iter_profile_log_dir"])
        if instance_cfg.get("iter_profile_log_dir")
        else None,
        command=[str(x) for x in instance_cfg["command"]],
        env={k: str(v) for k, v in instance_cfg["env"].items()},
        process=process,
    )


def terminate_runtimes(runtimes: Iterable[InstanceRuntime]) -> None:
    for rt in runtimes:
        proc = rt.process
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + 12
    while time.time() < deadline:
        if all(rt.process.poll() is not None for rt in runtimes):
            break
        time.sleep(0.5)

    for rt in runtimes:
        proc = rt.process
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def summarize_runtimes(runtimes: List[InstanceRuntime]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for rt in runtimes:
        summary.append(
            {
                "role": rt.role,
                "index": rt.index,
                "pid": rt.process.pid,
                "bind_host": rt.bind_host,
                "connect_host": rt.connect_host,
                "port": rt.port,
                "tp_size": rt.tp_size,
                "gpu_ids": rt.gpu_ids,
                "log_dir": str(rt.log_dir),
                "log_file": str(rt.log_file),
                "iter_profile_log_dir": (
                    str(rt.iter_profile_log_dir)
                    if rt.iter_profile_log_dir
                    else None
                ),
                "command": rt.command,
            }
        )
    return summary


def close_runtime_log_handles(runtimes: Iterable[InstanceRuntime]) -> None:
    for rt in runtimes:
        handle = getattr(rt.process, "_log_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass