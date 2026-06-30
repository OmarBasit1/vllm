# SABER expert-prediction recreation (Qwen1.5-MoE on vLLM)

Recreates the token -> cluster classifier from *Sparsity-Aware Scheduling for
Efficient MoE-Based LLM Serving on Edge Servers* (SABER), §5.1, on
`Qwen/Qwen1.5-MoE-A2.7B-Chat`. Scope is **Phases 1-5** (profiling + classifier +
accuracy); no scheduler / offloading / serving loop.

Routing is sourced from **vLLM's own gate + top-k** (not HuggingFace), captured
with forward hooks on each `Qwen2MoeSparseMoeBlock.gate` during **prefill-only,
batch-1** generation, so token i maps trivially to sequence position i. The
always-on shared expert never enters `router_logits`, so it is excluded from
paths automatically.

## Pipeline

| Phase | Script | Output (under `/export3/obasit/ClusterMoE/logs/qwen1.5_2.7B/saber/`) |
|---|---|---|
| 1 | `capture_paths.py` | `model_config.json`, `paths/<dataset>.parquet`, `capture_summary.json` |
| 2 | `valid_paths.py` | `valid_paths.parquet`, `valid_paths_report.json` |
| 3 | `cluster_paths.py` | `clusters.json`, `clusters_report.json` |
| 4 | `train_classifier.py` | `classifier.pt` (+ held-out acc, latency) |
| 5 | `evaluate.py` | `eval_report.md` (held-out + distribution-shift + latency) |

## Running

Always go through `run.sh`, which sets the conda env's `libstdc++` on the
library path (fixes a zmq `GLIBCXX` import error) and runs the V1 EngineCore
**in-process** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the capture hooks and
their state share memory with the driver.

```bash
# Smoke test (small): ~50 sequences per dataset
./run.sh capture_paths.py --max-seqs-per-dataset 50
./run.sh valid_paths.py
./run.sh cluster_paths.py
./run.sh train_classifier.py --epochs 15
./run.sh evaluate.py --epochs 15

# Full corpus run (handoff): drop the cap and tune pruning/clustering
./run.sh capture_paths.py                      # all sequences, all datasets
./run.sh valid_paths.py --tau 1e-4             # raise tau / --top-m if |V| is huge
./run.sh cluster_paths.py --Nc 128 --P 0.2 --alpha 1.0
./run.sh train_classifier.py --epochs 30
./run.sh evaluate.py --epochs 30
```

## Notes

- Corpus: Alpaca, XSUM, GLUE SST-2 + MNLI, MT-Bench (`corpus.py`). MT-Bench is
  loaded best-effort from a couple of HF sources; if unavailable it is skipped.
- `valid_paths.py` reports `|V|` and retained probability mass. Phase 3 is
  `O(|V|^2)`; if `|V|` is large, raise `--tau` or set `--top-m`.
- `cluster_paths.py` implements SABER Algorithm 1 exactly (distance
  `d(a,b)=|a∪b|-max(|a|,|b|)`, union centers, merge cost with the `alpha`
  probability penalty, and the `B / P / Nc` constraints). The optional
  similarity tree (for the out-of-scope Phase 6 scheduler) is skipped.
- Distribution-shift eval trains on XSUM and tests on the other datasets
  (paper Table 3 protocol).
