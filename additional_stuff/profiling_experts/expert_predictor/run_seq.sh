#!/usr/bin/env bash
# Orchestration for the past-token / past-layer experiments.
#
# Parallelism: for each window size, all 24 per-layer jobs are launched at once and
# pinned round-robin to the available GPUs via CUDA_VISIBLE_DEVICES (reuses the pattern
# from run_all.sh). The compact top-k store is built once and shared by both modes.
#
# Usage:
#   bash run_seq.sh --mode {tokens|layers} [options]
#
# Options:
#   --gpus N | --gpu-ids "0 1"   GPU selection (default: auto-detect, max 2)
#   --max-files N                Limit log files at build time (0 = all)
#   --skip-build                 Reuse existing store
#   --skip-hparam-search         Reuse cached shared_best_params.json (or defaults)
#   --skip-eval                  Skip final evaluation
#   --force-retrain              Retrain even if a checkpoint exists
#   --n-trials N                 Optuna trials per hparam-search cell (default: 20)

set -uo pipefail

PYTHON="${PYTHON:-/home/obasit/miniconda3/envs/vllm-moe/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B"
LOGS_DIR="$BASE/all_layers_pre_gating_logs"
# Shared compact store (built once, used by both modes).
STORE_DIR="$BASE/prediction-seq_store/dataset"

MODE=""
MAX_FILES=0
SKIP_BUILD=0
SKIP_HPARAM=0
SKIP_EVAL=0
FORCE_RETRAIN=0
N_TRIALS=20
NUM_GPUS=""
GPU_IDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)               MODE="$2";       shift 2 ;;
        --gpus)               NUM_GPUS="$2";   shift 2 ;;
        --gpu-ids)            GPU_IDS="$2";    shift 2 ;;
        --max-files)          MAX_FILES="$2";  shift 2 ;;
        --skip-build)         SKIP_BUILD=1;    shift ;;
        --skip-hparam-search) SKIP_HPARAM=1;   shift ;;
        --skip-eval)          SKIP_EVAL=1;     shift ;;
        --force-retrain)      FORCE_RETRAIN=1; shift ;;
        --n-trials)           N_TRIALS="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "tokens" && "$MODE" != "layers" ]]; then
    echo "ERROR: --mode must be 'tokens' or 'layers'"; exit 1
fi

# Mode-specific window sweep + outputs.
if [[ "$MODE" == "tokens" ]]; then
    WINDOWS=(1 2 3 4 5 6 7 8 9 10)
    MIDWIN=5
    OUT_ROOT="$BASE/prediction-past_tokens"
else
    WINDOWS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)
    MIDWIN=12
    OUT_ROOT="$BASE/prediction-past_layers"
fi
CKPT_BASE="$OUT_ROOT/checkpoints"
RESULTS_DIR="$OUT_ROOT/results"
LOG_DIR="$OUT_ROOT/joblogs"
SHARED_PARAMS="$CKPT_BASE/shared_best_params.json"
HSEARCH_LAYERS=(6 12 18)

# Resolve GPUs
if [[ -n "$GPU_IDS" ]]; then
    read -ra GPUS <<< "$GPU_IDS"
elif [[ -n "$NUM_GPUS" ]]; then
    GPUS=(); for (( i=0; i<NUM_GPUS; i++ )); do GPUS+=("$i"); done
else
    N_AVAIL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
    N_USE=$(( N_AVAIL < 2 ? N_AVAIL : 2 ))
    GPUS=(); for (( i=0; i<N_USE; i++ )); do GPUS+=("$i"); done
fi
N_GPU=${#GPUS[@]}
GPU_LIST=$(IFS=,; echo "${GPUS[*]}")
mkdir -p "$LOG_DIR" "$CKPT_BASE" "$RESULTS_DIR"

echo "============================================================"
echo " Sequence-feature pipeline — mode=$MODE"
echo " GPUs    : [$GPU_LIST] (${N_GPU})"
echo " Store   : $STORE_DIR"
echo " Outputs : $OUT_ROOT"
echo " Windows : ${WINDOWS[*]}"
echo "============================================================"

wait_all() {
    local pids=("$@"); local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then echo "  WARNING: job PID=$pid failed" >&2; failed=$((failed+1)); fi
    done
    [[ $failed -gt 0 ]] && echo "  $failed job(s) failed — see $LOG_DIR" >&2
    return 0
}
gpu_for() { echo "${GPUS[$(( $1 % N_GPU ))]}"; }

# ── Step 0: optuna ──
if ! "$PYTHON" -c "import optuna" 2>/dev/null; then
    echo "Installing optuna..."
    "$PYTHON" -m pip install optuna --quiet \
        && echo "  optuna installed." || echo "  WARNING: optuna install failed — grid fallback."
fi

# ── Step 1: build shared store (once) ──
if [[ $SKIP_BUILD -eq 0 && ! -f "$STORE_DIR/metadata.json" ]]; then
    echo "Step 1: Building shared store (GPU ${GPUS[0]})..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" "$SCRIPT_DIR/seq_dataset_builder.py" \
        --logs-dir "$LOGS_DIR" --out-dir "$STORE_DIR" --max-files "$MAX_FILES"
else
    echo "Step 1: Reusing store at $STORE_DIR"
fi

# ── Step 2: hparam search on representative cells ──
if [[ $SKIP_HPARAM -eq 0 ]]; then
    echo "Step 2: Hparam search (layers ${HSEARCH_LAYERS[*]} @ window $MIDWIN)..."
    PIDS=(); J=0
    for layer in "${HSEARCH_LAYERS[@]}"; do
        GPU=$(gpu_for $J)
        LOG="$LOG_DIR/hparam_layer${layer}.log"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/train_seq_predictor.py" \
            --mode "$MODE" --layer "$layer" --window "$MIDWIN" \
            --store-dir "$STORE_DIR" --ckpt-base "$CKPT_BASE" \
            --hparam-search --n-trials "$N_TRIALS" >"$LOG" 2>&1 &
        PIDS+=($!); J=$((J+1))
    done
    wait_all "${PIDS[@]}"
    # Pick the lowest-val-KL params among the search cells → shared_best_params.json
    "$PYTHON" - "$CKPT_BASE" "$MODE" "$MIDWIN" "$SHARED_PARAMS" "${HSEARCH_LAYERS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
ckpt_base, mode, midwin, out = sys.argv[1:5]
layers = [int(x) for x in sys.argv[5:]]
best = None
for L in layers:
    p = Path(ckpt_base) / f"{mode}_layer_{L:02d}_win_{int(midwin):02d}" / "best_params.json"
    if p.exists():
        d = json.loads(p.read_text())
        if best is None or d.get("best_val_kl", 1e9) < best.get("best_val_kl", 1e9):
            best = d
if best is not None:
    Path(out).write_text(json.dumps(best, indent=2))
    print(f"  shared params → {out}: {best}")
else:
    print("  WARNING: no best_params.json found; sweep will use CLI defaults")
PYEOF
else
    echo "Step 2: Skipping hparam search."
fi

SHARED_FLAG=""
[[ -f "$SHARED_PARAMS" ]] && SHARED_FLAG="--shared-params $SHARED_PARAMS"
FORCE_FLAG=""
[[ $FORCE_RETRAIN -eq 1 ]] && FORCE_FLAG="--force-retrain"

# ── Step 3: sweep all windows × all layers ──
echo "Step 3: Training sweep..."
for window in "${WINDOWS[@]}"; do
    echo "  window=$window — 24 layers across [$GPU_LIST]..."
    PIDS=(); J=0
    for (( layer=0; layer<24; layer++ )); do
        GPU=$(gpu_for $J)
        LOG="$LOG_DIR/train_win${window}_layer${layer}.log"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/train_seq_predictor.py" \
            --mode "$MODE" --layer "$layer" --window "$window" \
            --store-dir "$STORE_DIR" --ckpt-base "$CKPT_BASE" \
            $SHARED_FLAG $FORCE_FLAG >"$LOG" 2>&1 &
        PIDS+=($!); J=$((J+1))
    done
    wait_all "${PIDS[@]}"
    echo "  window=$window done."
done

# ── Step 4: evaluate ──
if [[ $SKIP_EVAL -eq 0 ]]; then
    echo "Step 4: Evaluating on test split..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" "$SCRIPT_DIR/evaluate_seq.py" \
        --mode "$MODE" --store-dir "$STORE_DIR" \
        --ckpt-base "$CKPT_BASE" --out-dir "$RESULTS_DIR"
fi

echo "============================================================"
echo " Done (mode=$MODE)."
echo "  Checkpoints: $CKPT_BASE"
echo "  Results    : $RESULTS_DIR/summary.csv + heatmaps + marginal_by_window.png"
echo "  Job logs   : $LOG_DIR"
echo "============================================================"
