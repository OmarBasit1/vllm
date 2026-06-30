"""Shared constants and paths for the SABER expert-prediction recreation.

Pipeline (Phases 1-5; no scheduler/offloading):
  capture_paths -> valid_paths -> cluster_paths -> train_classifier -> evaluate

All routing is sourced from vLLM's own gate + top-k (Qwen2MoE architecture).
Run every script in the ``vllm-moe`` conda env with the env's libstdc++ on the
library path, e.g.::

    LD_LIBRARY_PATH=/home/obasit/miniconda3/envs/vllm-moe/lib \
      /home/obasit/miniconda3/envs/vllm-moe/bin/python capture_paths.py ...

(see ``run.sh`` which wraps this).
"""
from __future__ import annotations

from pathlib import Path

# Cached chat variant; router architecture is identical to the base model and
# the SABER corpus is instruction/chat style.
MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"

# /export3 has the free space (/export2 is ~92% full).
OUT_ROOT = Path("/export3/obasit/ClusterMoE/logs/qwen1.5_2.7B/saber")

PATHS_DIR = OUT_ROOT / "paths"            # paths/<dataset>.parquet
MODEL_CONFIG = OUT_ROOT / "model_config.json"
VALID_PATHS = OUT_ROOT / "valid_paths.parquet"
CLUSTERS = OUT_ROOT / "clusters.json"
CLASSIFIER = OUT_ROOT / "classifier.pt"
EVAL_REPORT = OUT_ROOT / "eval_report.md"

# The five corpora in the SABER NLP mix.
DATASETS = ["alpaca", "xsum", "sst2", "mnli", "mtbench"]
