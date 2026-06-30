#!/usr/bin/env python
"""Phase 2 - valid-path identification (SABER paper section 5.1.1).

Aggregate captured per-token activation paths into distinct paths with their
frequency Freq(p) and probability Pr(p)=Freq(p)/sum. Prune Pr(p) < tau and,
for tractability of the O(|V|^2) clustering in Phase 3, optionally cap to the
top-M paths by frequency. Report |V| before/after and the retained mass.

A path is canonicalized as: for each MoE layer (in capture order), the sorted
tuple of its k selected expert ids; concatenated layer-major. This is the
frozenset {(layer, expert)} representation used throughout the pipeline
(layer = slot_index // k).

Inputs : common.PATHS_DIR/*.parquet, common.MODEL_CONFIG
Output : common.VALID_PATHS (path_id, freq, pr, experts[]) + valid_paths_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def canonical_path(flat_experts: np.ndarray, k: int, n_slots: int) -> tuple:
    """Return the canonical (per-layer-sorted) tuple for a flat experts row."""
    layers = flat_experts.reshape(n_slots // k, k)
    layers = np.sort(layers, axis=1)
    return tuple(int(x) for x in layers.reshape(-1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tau", type=float, default=1e-4, help="prune Pr(p) < tau")
    ap.add_argument("--top-m", type=int, default=0,
                    help="after tau-pruning, cap to top-M by frequency (0 = no cap)")
    ap.add_argument("--datasets", nargs="*", default=common.DATASETS)
    args = ap.parse_args()

    mc = json.loads(common.MODEL_CONFIG.read_text())
    k = mc["num_experts_per_tok"]
    n_slots = mc["active_slots_per_token"]

    counter: Counter = Counter()
    total = 0
    for ds in args.datasets:
        p = common.PATHS_DIR / f"{ds}.parquet"
        if not p.exists():
            print(f"  skip missing {p}", flush=True)
            continue
        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(batch_size=100_000, columns=["experts"]):
            exp = np.asarray(batch.column("experts").to_pylist(), dtype=np.int16)
            for row in exp:
                counter[canonical_path(row, k, n_slots)] += 1
                total += 1
        print(f"  {ds}: cumulative tokens={total}, distinct paths={len(counter)}", flush=True)

    if total == 0:
        raise SystemExit("no tokens found; run capture_paths.py first")

    items = counter.most_common()  # sorted by freq desc
    freqs = np.array([c for _, c in items], dtype=np.int64)
    prs = freqs / total
    n_before = len(items)

    keep = prs >= args.tau
    kept_items = [items[i] for i in range(n_before) if keep[i]]
    if args.top_m and len(kept_items) > args.top_m:
        kept_items = kept_items[: args.top_m]

    kept_freq = np.array([c for _, c in kept_items], dtype=np.int64)
    kept_pr = kept_freq / total
    retained_mass = float(kept_freq.sum() / total)

    # Write valid paths.
    schema = pa.schema([
        ("path_id", pa.int32()),
        ("freq", pa.int64()),
        ("pr", pa.float64()),
        ("experts", pa.list_(pa.int16(), n_slots)),
    ])
    table = pa.table(
        {
            "path_id": pa.array(list(range(len(kept_items))), pa.int32()),
            "freq": pa.array(kept_freq, pa.int64()),
            "pr": pa.array(kept_pr, pa.float64()),
            "experts": pa.array([list(p) for p, _ in kept_items],
                                pa.list_(pa.int16(), n_slots)),
        },
        schema=schema,
    )
    pq.write_table(table, str(common.VALID_PATHS))

    report = {
        "total_tokens": total,
        "distinct_paths_before": n_before,
        "tau": args.tau,
        "top_m": args.top_m,
        "valid_paths_after": len(kept_items),
        "retained_probability_mass": retained_mass,
        "max_path_mem": n_slots,
        "note": "expect large |V| (top-%d x %d layers); prune/cap as needed for O(|V|^2)"
                % (k, n_slots // k),
    }
    (common.OUT_ROOT / "valid_paths_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== valid-path report ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    if len(kept_items) > 5000:
        print("\nWARNING: |V| > 5000; Phase 3 is O(|V|^2). Raise --tau or set --top-m.",
              flush=True)


if __name__ == "__main__":
    main()
