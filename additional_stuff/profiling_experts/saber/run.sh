#!/usr/bin/env bash
# Wrapper that runs a SABER pipeline script in the vllm-moe conda env with the
# env's libstdc++ on the library path (fixes the zmq GLIBCXX import error).
#
#   ./run.sh capture_paths.py --max-seqs-per-dataset 50
#   ./run.sh valid_paths.py
#   ./run.sh cluster_paths.py
#   ./run.sh train_classifier.py
#   ./run.sh evaluate.py
set -euo pipefail
ENV=/home/obasit/miniconda3/envs/vllm-moe
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run the V1 EngineCore in-process so apply_model executes in this process
# (lets us register Python forward hooks and read their state back directly).
exec env LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}" \
     VLLM_ENABLE_V1_MULTIPROCESSING=0 \
     "$ENV/bin/python" "$HERE/$@"
