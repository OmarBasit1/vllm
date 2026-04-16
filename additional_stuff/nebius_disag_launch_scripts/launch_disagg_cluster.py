#!/usr/bin/env python3
"""Config-driven launcher for disaggregated prefill/decode vLLM instances."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from launcher_lib import (
    build_instances_and_commands,
    close_runtime_log_handles,
    dump_json,
    launch_instance,
    load_json,
    resolve_path,
    summarize_runtimes,
    terminate_runtimes,
    utc_timestamp,
    wait_for_ready_instances,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id. Default: auto timestamped id",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands and exit without launching processes",
    )
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="Launch only prefill/decode instances and skip proxy startup",
    )
    return parser.parse_args()


def _proxy_connect_host(proxy_cfg: Dict[str, Any]) -> str:
    if proxy_cfg.get("connect_host"):
        return str(proxy_cfg["connect_host"])
    host = str(proxy_cfg.get("host", "127.0.0.1"))
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host


def _build_proxy_instance_cfg(
    cfg: Dict[str, Any],
    config_path: Path,
    run_dir: Path,
    prefill_endpoints: List[str],
    decode_endpoints: List[str],
) -> Dict[str, Any]:
    proxy_cfg = cfg.get("proxy", {})
    script_path = resolve_path(
        str(
            cfg.get(
                "proxy_script",
                str(Path(__file__).with_name("disagg_proxy_pool.py")),
            )
        ),
        config_path.parent,
    )

    python_executable = str(cfg.get("python_executable", sys.executable))
    bind_host = str(proxy_cfg.get("host", "127.0.0.1"))
    connect_host = _proxy_connect_host(proxy_cfg)
    port = int(proxy_cfg.get("port", 9000))

    proxy_log_dir = run_dir / "proxy"
    proxy_log_dir.mkdir(parents=True, exist_ok=True)
    proxy_log_file = proxy_log_dir / "proxy.log"

    command = [
        python_executable,
        str(script_path),
        "--host",
        bind_host,
        "--port",
        str(port),
        "--prefill-endpoints",
        ",".join(prefill_endpoints),
        "--decode-endpoints",
        ",".join(decode_endpoints),
        "--request-timeout-sec",
        str(int(proxy_cfg.get("request_timeout_sec", 600))),
    ]

    return {
        "role": "proxy",
        "index": 0,
        "bind_host": bind_host,
        "connect_host": connect_host,
        "port": port,
        "tp_size": 0,
        "gpu_ids": [],
        "log_dir": str(proxy_log_dir),
        "log_file": str(proxy_log_file),
        "moe_log_dir": None,
        "command": command,
        "env": dict(os.environ),
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_cfg = load_json(config_path)

    run_id = args.run_id or str(raw_cfg.get("run_id") or f"disagg_{utc_timestamp()}")
    main_log_dir = resolve_path(str(raw_cfg["main_log_dir"]), config_path.parent)
    run_dir = main_log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg, instance_cfgs = build_instances_and_commands(
        raw_cfg,
        config_path=config_path,
        run_dir=run_dir,
    )

    if args.dry_run:
        print("Dry run: resolved launch plan")
        for inst in instance_cfgs:
            print(
                f"- {inst['role']}[{inst['index']}] "
                f"host={inst['bind_host']} port={inst['port']} gpus={inst['gpu_ids']} "
                f"tp={inst['tp_size']}"
            )
            print("  command:", " ".join(inst["command"]))
        return 0

    health_timeout = float(cfg.get("launcher", {}).get("health_check_timeout_sec", 900))
    poll_interval = float(cfg.get("launcher", {}).get("health_check_interval_sec", 2))
    working_dir = Path(cfg["working_dir"])

    runtimes = []
    proxy_runtime = None

    try:
        print(f"Launching {len(instance_cfgs)} vLLM instances...")
        for inst_cfg in instance_cfgs:
            runtime = launch_instance(inst_cfg, working_dir=working_dir)
            runtimes.append(runtime)
            print(
                f"[launch] {runtime.role}[{runtime.index}] pid={runtime.process.pid} "
                f"host={runtime.bind_host}:{runtime.port} gpus={runtime.gpu_ids} tp={runtime.tp_size}"
            )

        wait_for_ready_instances(
            runtimes,
            timeout_sec=health_timeout,
            poll_interval_sec=poll_interval,
        )

        if not args.skip_proxy:
            prefill_endpoints = [
                f"{rt.connect_host}:{rt.port}" for rt in runtimes if rt.role == "prefill"
            ]
            decode_endpoints = [
                f"{rt.connect_host}:{rt.port}" for rt in runtimes if rt.role == "decode"
            ]
            if not prefill_endpoints or not decode_endpoints:
                raise ValueError("Need at least one prefill and one decode instance")

            proxy_cfg = _build_proxy_instance_cfg(
                cfg=cfg,
                config_path=config_path,
                run_dir=run_dir,
                prefill_endpoints=prefill_endpoints,
                decode_endpoints=decode_endpoints,
            )
            proxy_runtime = launch_instance(proxy_cfg, working_dir=working_dir)
            print(
                f"[launch] proxy pid={proxy_runtime.process.pid} "
                f"host={proxy_runtime.bind_host}:{proxy_runtime.port}"
            )
            wait_for_ready_instances(
                [proxy_runtime],
                timeout_sec=health_timeout,
                poll_interval_sec=poll_interval,
            )

        state = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "working_dir": str(working_dir),
            "config_path": str(config_path),
            "model": cfg["model"],
            "proxy_enabled": not args.skip_proxy,
            "instances": summarize_runtimes(runtimes),
            "proxy": summarize_runtimes([proxy_runtime])[0] if proxy_runtime else None,
        }
        state_file = run_dir / "cluster_state.json"
        dump_json(state_file, state)

        print("\nCluster launch successful.")
        print(f"Run directory: {run_dir}")
        print(f"State file: {state_file}")
        print("Processes are running in the background; inspect per-instance logs under the run directory.")
        return 0

    except Exception as exc:
        all_runtimes = list(runtimes)
        if proxy_runtime is not None:
            all_runtimes.append(proxy_runtime)
        terminate_runtimes(all_runtimes)
        print(f"Launch failed: {exc}", file=sys.stderr)
        return 1

    finally:
        all_runtimes = list(runtimes)
        if proxy_runtime is not None:
            all_runtimes.append(proxy_runtime)
        close_runtime_log_handles(all_runtimes)


if __name__ == "__main__":
    raise SystemExit(main())
