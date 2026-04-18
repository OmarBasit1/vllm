#!/usr/bin/env python3
"""Stop disaggregated vLLM instances, proxy, and optional aiPerf benchmarks."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from launcher_lib import load_json, resolve_path


STATE_FILE_NAME = "cluster_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Cluster config path used to resolve main_log_dir/run_id",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run id under main_log_dir to stop",
    )
    parser.add_argument(
        "--main-log-dir",
        default=None,
        help="Override main_log_dir when resolving state files",
    )
    parser.add_argument(
        "--state-file",
        action="append",
        default=[],
        help="Explicit cluster_state.json path (repeatable)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Do not stop benchmark processes (aiperf / run_aiperf_benchmark)",
    )
    parser.add_argument(
        "--no-pattern-fallback",
        action="store_true",
        help="Only stop PIDs listed in state files; skip process-pattern fallback",
    )
    parser.add_argument(
        "--grace-sec",
        type=float,
        default=10.0,
        help="Grace period between SIGTERM and SIGKILL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned process stops without sending signals",
    )
    return parser.parse_args()


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _send_signal(pid: int, sig: signal.Signals) -> bool:
    if not _is_pid_alive(pid):
        return False

    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return False
    except PermissionError:
        pgid = None

    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        except OSError:
            pass

    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def _wait_for_exit(pid: int, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(0.2)
    return not _is_pid_alive(pid)


def _stop_pid(pid: int, *, label: str, grace_sec: float, dry_run: bool) -> Tuple[bool, str]:
    if not _is_pid_alive(pid):
        return False, f"[skip] {label}: pid={pid} is not running"

    if dry_run:
        return True, f"[dry-run] would stop {label}: pid={pid}"

    sent_term = _send_signal(pid, signal.SIGTERM)
    if not sent_term:
        if _is_pid_alive(pid):
            return False, f"[error] failed to send SIGTERM to {label}: pid={pid}"
        return False, f"[skip] {label}: pid={pid} exited before SIGTERM"

    if _wait_for_exit(pid, timeout_sec=grace_sec):
        return True, f"[stopped] {label}: pid={pid} (SIGTERM)"

    sent_kill = _send_signal(pid, signal.SIGKILL)
    if not sent_kill and _is_pid_alive(pid):
        return False, f"[error] failed to send SIGKILL to {label}: pid={pid}"

    if _wait_for_exit(pid, timeout_sec=2.0):
        return True, f"[stopped] {label}: pid={pid} (SIGKILL)"

    return False, f"[error] process still alive after SIGKILL for {label}: pid={pid}"


def _find_latest_state_file(main_log_dir: Path) -> Optional[Path]:
    candidates = sorted(
        main_log_dir.glob(f"*/{STATE_FILE_NAME}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _append_unique_path(paths: List[Path], candidate: Path) -> None:
    if candidate not in paths:
        paths.append(candidate)


def _resolve_state_files(args: argparse.Namespace) -> List[Path]:
    resolved: List[Path] = []

    for state_file in args.state_file:
        path = Path(state_file).expanduser().resolve()
        if path.exists():
            _append_unique_path(resolved, path)
        else:
            print(f"[warn] explicit state file not found: {path}")

    config_path: Optional[Path] = None
    main_log_dir: Optional[Path] = None

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        cfg = load_json(config_path)
        main_log_dir_text = args.main_log_dir or cfg.get("main_log_dir")
        if not main_log_dir_text:
            raise ValueError("Unable to resolve main_log_dir from --config; pass --main-log-dir")
        main_log_dir = resolve_path(str(main_log_dir_text), config_path.parent)

        run_id = args.run_id or cfg.get("run_id")
        if run_id:
            state_file = (main_log_dir / str(run_id) / STATE_FILE_NAME).resolve()
            if state_file.exists():
                _append_unique_path(resolved, state_file)
            else:
                print(f"[warn] state file not found for run_id '{run_id}': {state_file}")
        else:
            latest = _find_latest_state_file(main_log_dir)
            if latest is not None:
                _append_unique_path(resolved, latest.resolve())
            else:
                print(f"[warn] no {STATE_FILE_NAME} found under {main_log_dir}")

    elif args.main_log_dir:
        main_log_dir = Path(args.main_log_dir).expanduser().resolve()
        if args.run_id:
            state_file = (main_log_dir / str(args.run_id) / STATE_FILE_NAME).resolve()
            if state_file.exists():
                _append_unique_path(resolved, state_file)
            else:
                print(f"[warn] state file not found for run_id '{args.run_id}': {state_file}")
        else:
            latest = _find_latest_state_file(main_log_dir)
            if latest is not None:
                _append_unique_path(resolved, latest.resolve())
            else:
                print(f"[warn] no {STATE_FILE_NAME} found under {main_log_dir}")

    return resolved


def _read_process_table() -> Dict[int, str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    table: Dict[int, str] = {}
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, args = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        table[pid] = args
    return table


def _matches_state_target(target: Dict[str, Any], cmdline: str) -> bool:
    role = str(target.get("role", ""))
    port = target.get("port")
    port_text = str(port) if port is not None else ""

    if role == "proxy":
        if "disagg_proxy_pool.py" not in cmdline:
            return False
        return port_text in cmdline if port_text else True

    if role in ("prefill", "decode"):
        if "vllm" not in cmdline:
            return False
        return port_text in cmdline if port_text else True

    return True


def _collect_state_targets(state_files: Iterable[Path]) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for state_file in state_files:
        try:
            state = load_json(state_file)
        except Exception as exc:
            print(f"[warn] failed to read state file {state_file}: {exc}")
            continue

        run_id = state.get("run_id", "<unknown>")
        for entry in state.get("instances", []):
            if not isinstance(entry, dict):
                continue
            pid = entry.get("pid")
            if pid is None:
                continue
            targets.append(
                {
                    "pid": int(pid),
                    "role": str(entry.get("role", "instance")),
                    "index": entry.get("index"),
                    "port": entry.get("port"),
                    "source": str(state_file),
                    "run_id": str(run_id),
                }
            )

        proxy = state.get("proxy")
        if isinstance(proxy, dict) and proxy.get("pid") is not None:
            targets.append(
                {
                    "pid": int(proxy["pid"]),
                    "role": "proxy",
                    "index": proxy.get("index"),
                    "port": proxy.get("port"),
                    "source": str(state_file),
                    "run_id": str(run_id),
                }
            )

    dedup: Dict[int, Dict[str, Any]] = {}
    for target in targets:
        dedup[target["pid"]] = target
    return list(dedup.values())


def _match_process_patterns(
    process_table: Dict[int, str],
    *,
    include_benchmark: bool,
) -> List[Tuple[int, str, str]]:
    pattern_sets: List[Tuple[str, List[str]]] = [
        ("vllm", ["vllm serve", "vllm.entrypoints.openai.api_server"]),
        ("proxy", ["disagg_proxy_pool.py"]),
    ]
    if include_benchmark:
        pattern_sets.append(
            (
                "benchmark",
                [
                    "aiperf profile",
                    "run_aiperf_benchmark.py",
                    "run_aiperf_benchmark.sh",
                ],
            )
        )

    current_pid = os.getpid()
    matches: Dict[int, Tuple[str, str]] = {}
    for kind, patterns in pattern_sets:
        for pid, cmdline in process_table.items():
            if pid == current_pid:
                continue
            if any(pattern in cmdline for pattern in patterns):
                matches[pid] = (kind, cmdline)

    return [(pid, kind, cmdline) for pid, (kind, cmdline) in matches.items()]


def main() -> int:
    args = parse_args()

    try:
        state_files = _resolve_state_files(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if state_files:
        print("Using state files:")
        for state_file in state_files:
            print(f"  - {state_file}")
    else:
        print("No state files resolved. Falling back to process-pattern matching.")

    targets = _collect_state_targets(state_files)
    if targets:
        print(f"Resolved {len(targets)} process target(s) from state files.")

    process_table = _read_process_table()
    stopped = 0
    failed = 0
    seen_pids = set()

    for target in targets:
        pid = int(target["pid"])
        seen_pids.add(pid)

        cmdline = process_table.get(pid)
        role = str(target.get("role", "process"))
        idx = target.get("index")
        port = target.get("port")
        label = f"{role}[{idx}] run={target.get('run_id')} port={port}"

        if cmdline and not _matches_state_target(target, cmdline):
            print(
                f"[warn] state target mismatch; skipping pid={pid} for {label}\n"
                f"       live cmdline: {cmdline}"
            )
            continue

        ok, message = _stop_pid(
            pid,
            label=label,
            grace_sec=max(args.grace_sec, 0.0),
            dry_run=args.dry_run,
        )
        print(message)
        if ok:
            stopped += 1
        elif "[error]" in message:
            failed += 1

    if not args.no_pattern_fallback:
        process_table = _read_process_table()
        fallback_matches = _match_process_patterns(
            process_table,
            include_benchmark=not args.skip_benchmark,
        )

        fallback_matches = [m for m in fallback_matches if m[0] not in seen_pids]
        if fallback_matches:
            print(f"Pattern fallback found {len(fallback_matches)} additional process(es).")

        for pid, kind, cmdline in sorted(fallback_matches, key=lambda x: x[0]):
            label = f"{kind} fallback"
            ok, message = _stop_pid(
                pid,
                label=label,
                grace_sec=max(args.grace_sec, 0.0),
                dry_run=args.dry_run,
            )
            print(message)
            if ok:
                stopped += 1
            elif "[error]" in message:
                failed += 1
            if not args.dry_run:
                print(f"       cmdline: {cmdline}")

    elif not targets:
        print("[warn] no state targets found and --no-pattern-fallback was set")

    if args.dry_run:
        print(f"Dry run complete. Planned stops: {stopped}")
        return 0

    if stopped == 0 and failed == 0:
        print("No matching running processes were found.")
    else:
        print(f"Stop complete. stopped={stopped} failed={failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
