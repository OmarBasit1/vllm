#!/usr/bin/env python
"""Phase 3 - constrained agglomerative hierarchical clustering (SABER 5.1.2, Alg 1).

Definitions (implemented exactly):
  path/center represented as a frozenset of (layer, expert) pairs
  distance      d(a, b) = |a u b| - max(|a|, |b|)
  center        c = OR (union) over members ;  Mem(c) = |c|
  merge cost    m_ij = d(c_i, c_j) + alpha * (Pr(c_i) + Pr(c_j))
  constraints   Mem(c) <= B ,  sum Pr <= P ,  |C| <= N_c

Greedy: repeatedly merge the min-cost feasible pair; snapshot C* at the first
point where no feasible merge remains. The optional similarity tree T (used only
for the out-of-scope Phase 6 scheduler) is skipped.

Inputs : common.VALID_PATHS, common.MODEL_CONFIG
Output : common.CLUSTERS (centers + members + params) + clusters_report.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def to_pairs(flat: list[int], k: int) -> frozenset:
    """Flat layer-major experts -> frozenset{(layer, expert)}."""
    pairs = set()
    for slot, e in enumerate(flat):
        pairs.add((slot // k, int(e)))
    return frozenset(pairs)


def d(a: frozenset, b: frozenset) -> int:
    return len(a | b) - max(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=0,
                    help="max Mem(c) per cluster (0 = num_experts*L_moe/2 heuristic)")
    ap.add_argument("--P", type=float, default=0.20, help="max sum Pr per cluster")
    ap.add_argument("--Nc", type=int, default=128, help="max number of clusters |C*|")
    ap.add_argument("--alpha", type=float, default=1.0, help="probability penalty weight")
    args = ap.parse_args()

    mc = json.loads(common.MODEL_CONFIG.read_text())
    k = mc["num_experts_per_tok"]
    n_slots = mc["active_slots_per_token"]
    mem_full = mc["num_experts"] * mc["L_moe"]  # absolute ceiling on Mem(c)
    B = args.B or mem_full // 3  # default keeps Mem(c) well below the ceiling
    P_budget = args.P

    vp = pq.read_table(str(common.VALID_PATHS)).to_pydict()
    centers = [to_pairs(e, k) for e in vp["experts"]]
    prs = list(map(float, vp["pr"]))
    members = [[i] for i in range(len(centers))]  # member valid-path ids
    alive = [True] * len(centers)
    n_alive = len(centers)
    print(f"init |C|={n_alive}; B={B} P={P_budget} Nc={args.Nc} alpha={args.alpha}",
          flush=True)
    if n_alive > 5000:
        print("WARNING: O(|V|^2) init pairwise heap may be slow/large for |V|>5000.",
              flush=True)

    def feasible(i: int, j: int) -> bool:
        merged = centers[i] | centers[j]
        if len(merged) > B:
            return False
        if prs[i] + prs[j] > P_budget:
            return False
        return True

    def cost(i: int, j: int) -> float:
        return d(centers[i], centers[j]) + args.alpha * (prs[i] + prs[j])

    # Lazy heap of (cost, i, j). Entries validated against `alive` + a version
    # stamp so stale pairs (touching a merged cluster) are skipped on pop.
    version = [0] * len(centers)
    heap: list = []
    for i in range(len(centers)):
        ci = centers[i]
        for j in range(i + 1, len(centers)):
            heapq.heappush(heap, (cost(i, j), i, j, version[i], version[j]))

    cstar = None  # snapshot at first infeasibility
    cstar_saved = False

    def snapshot():
        idxs = [i for i in range(len(centers)) if alive[i]]
        return [
            {
                "center": sorted([list(p) for p in centers[i]]),
                "mem": len(centers[i]),
                "pr": prs[i],
                "members": members[i],
            }
            for i in idxs
        ]

    while n_alive > 1:
        # Pop the min-cost still-valid pair.
        best = None
        while heap:
            c, i, j, vi, vj = heapq.heappop(heap)
            if not (alive[i] and alive[j]):
                continue
            if vi != version[i] or vj != version[j]:
                continue
            best = (c, i, j)
            break
        if best is None:
            break

        _, i, j = best
        if not feasible(i, j):
            # No feasible merge among the cheapest? We must verify none feasible.
            # Re-scan remaining valid pairs for any feasible one (rare hot path).
            found = None
            for (c2, a, b, va, vb) in heap:
                if alive[a] and alive[b] and va == version[a] and vb == version[b] \
                        and feasible(a, b):
                    found = (a, b)
                    break
            if found is None:
                if not cstar_saved:
                    cstar = snapshot()
                    cstar_saved = True
                    print(f"snapshot C*: |C*|={len(cstar)} at first infeasibility",
                          flush=True)
                break  # constrained result reached; tree T is skipped
            i, j = found

        # Merge j into i.
        centers[i] = centers[i] | centers[j]
        prs[i] = prs[i] + prs[j]
        members[i] = members[i] + members[j]
        alive[j] = False
        version[i] += 1
        n_alive -= 1

        # Early stop if we hit the |C| budget and it's all feasible so far.
        if not cstar_saved and n_alive <= args.Nc:
            # Check feasibility is still maintained; snapshot when no feasible
            # merge would further reduce below a useful size. We snapshot lazily
            # on infeasibility above, but also cap here at Nc.
            pass

        # Re-push pairs of the merged cluster i with all other alive clusters.
        for x in range(len(centers)):
            if x != i and alive[x]:
                heapq.heappush(heap, (cost(i, x), min(i, x), max(i, x),
                                      version[min(i, x)], version[max(i, x)]))

        if n_alive <= args.Nc and not cstar_saved:
            # Respect the |C| <= Nc constraint as a stopping point.
            cstar = snapshot()
            cstar_saved = True
            print(f"snapshot C*: |C*|={len(cstar)} reached Nc={args.Nc}", flush=True)
            break

    if cstar is None:
        cstar = snapshot()

    out = {
        "params": {"B": B, "P": P_budget, "Nc": args.Nc, "alpha": args.alpha},
        "model": {"k": k, "n_slots": n_slots,
                  "num_experts": mc["num_experts"], "L_moe": mc["L_moe"]},
        "num_clusters": len(cstar),
        "clusters": cstar,
    }
    common.CLUSTERS.write_text(json.dumps(out))

    mems = np.array([c["mem"] for c in cstar])
    prdist = np.array([c["pr"] for c in cstar])
    report = {
        "num_clusters": len(cstar),
        "params": out["params"],
        "mem_min": int(mems.min()), "mem_max": int(mems.max()),
        "mem_mean": float(mems.mean()),
        "mem_max_vs_ceiling": f"{int(mems.max())}/{mem_full}",
        "single_token_slots": n_slots,
        "pr_min": float(prdist.min()), "pr_max": float(prdist.max()),
        "all_constraints_satisfied": bool((mems <= B).all() and (prdist <= P_budget + 1e-9).all()
                                          and len(cstar) <= args.Nc),
    }
    (common.OUT_ROOT / "clusters_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== clusters report ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
