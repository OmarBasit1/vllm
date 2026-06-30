#!/usr/bin/env bash
# Full-corpus SABER run: capture -> valid paths -> cluster -> train -> evaluate.
# Designed to run detached overnight. Each stage logs with timestamps; the run
# aborts if any stage fails (set -e).
#
# Launch detached so it survives SSH disconnect:
#   setsid bash full_run.sh >/export3/.../saber/full_run.log 2>&1 </dev/null &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Per-dataset sequence cap. mtbench has only ~80; others are capped here so the
# batch-1 prefill capture finishes overnight. Raise for more coverage.
MAX_SEQS="${MAX_SEQS:-10000}"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }
stage() { echo; echo "===== [$(stamp)] $* ====="; }

stage "START full SABER run (MAX_SEQS=$MAX_SEQS)"

stage "Phase 1: capture_paths"
./run.sh capture_paths.py --max-seqs-per-dataset "$MAX_SEQS" \
    --max-len 1024 --max-model-len 2048

stage "Phase 2: valid_paths (tau=1e-5, cap top-m=4000 for O(|V|^2))"
./run.sh valid_paths.py --tau 1e-5 --top-m 4000

stage "Phase 3: cluster_paths (Nc=128, P=0.02, alpha=1.0)"
./run.sh cluster_paths.py --Nc 128 --P 0.02 --alpha 1.0

stage "Phase 4: train_classifier (30 epochs)"
./run.sh train_classifier.py --epochs 30

stage "Phase 5: evaluate (30 epochs)"
./run.sh evaluate.py --epochs 30

stage "DONE. See eval_report.md"
