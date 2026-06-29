#!/usr/bin/env bash
# Orchestration for the combined past-layer + cross-gating-offset experiment.
#
# Two configs (window 5): offset {1,2,3,4} and offset {1}. Per config: small hparam
# search on representative layers → train all offset-valid layers in parallel across both
# GPUs → joint evaluation. Reuses the existing offset store and sequence store (no build).
#
# Usage:
#   bash run_combined.sh [--gpus N | --gpu-ids "0 1"] [--n-trials N]
#                        [--skip-hparam-search] [--skip-eval] [--force-retrain]

set -uo pipefail

PYTHON="${PYTHON:-/home/obasit/miniconda3/envs/vllm-moe/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B"
SEQ_STORE="$BASE/prediction-seq_store/dataset"
OFFSET_STORE="$BASE/predictions-cross_gating/expert_predictor_dataset"

OUT_ROOT="$BASE/prediction-combined"
CKPT_BASE="$OUT_ROOT/checkpoints"
RESULTS_DIR="$OUT_ROOT/results"
LOG_DIR="$OUT_ROOT/joblogs"

WINDOW=5
HSEARCH_LAYERS=(6 12 18)
# Configs: space-separated offset combos.
CONFIGS=("1 2 3 4" "1")

N_TRIALS=20
SKIP_HPARAM=0
SKIP_EVAL=0
FORCE_RETRAIN=0
NUM_GPUS=""
GPU_IDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)               NUM_GPUS="$2";   shift 2 ;;
        --gpu-ids)            GPU_IDS="$2";    shift 2 ;;
        --n-trials)           N_TRIALS="$2";   shift 2 ;;
        --skip-hparam-search) SKIP_HPARAM=1;   shift ;;
        --skip-eval)          SKIP_EVAL=1;     shift ;;
        --force-retrain)      FORCE_RETRAIN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
echo " Combined past-layer(win$WINDOW) + offset pipeline"
echo " GPUs        : [$GPU_LIST] (${N_GPU})"
echo " Seq store   : $SEQ_STORE"
echo " Offset store: $OFFSET_STORE"
echo " Outputs     : $OUT_ROOT"
echo " Configs     : ${CONFIGS[*]/#/offset=}"
echo "============================================================"

if [[ ! -f "$SEQ_STORE/metadata.json" ]]; then
    echo "ERROR: sequence store missing at $SEQ_STORE"; exit 1
fi
if [[ ! -f "$OFFSET_STORE/metadata.json" ]]; then
    echo "ERROR: offset store missing at $OFFSET_STORE"; exit 1
fi

wait_all() {
    local pids=("$@"); local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then echo "  WARNING: job PID=$pid failed" >&2; failed=$((failed+1)); fi
    done
    [[ $failed -gt 0 ]] && echo "  $failed job(s) failed — see $LOG_DIR" >&2
    return 0
}
gpu_for() { echo "${GPUS[$(( $1 % N_GPU ))]}"; }
combo_tag() { echo "${1// /_}"; }

# optuna
if ! "$PYTHON" -c "import optuna" 2>/dev/null; then
    echo "Installing optuna..."
    "$PYTHON" -m pip install optuna --quiet \
        && echo "  optuna installed." || echo "  WARNING: optuna install failed — grid fallback."
fi

FORCE_FLAG=""; [[ $FORCE_RETRAIN -eq 1 ]] && FORCE_FLAG="--force-retrain"

for config in "${CONFIGS[@]}"; do
    TAG=$(combo_tag "$config")
    MINL=$(echo "$config" | tr ' ' '\n' | sort -n | tail -1)
    SHARED_PARAMS="$CKPT_BASE/shared_best_params_off_${TAG}.json"
    echo ""
    echo ">>> Config offset={$config}  (valid layers ${MINL}..23)"

    # ── hparam search (representative layers, parallel) ──
    if [[ $SKIP_HPARAM -eq 0 ]]; then
        echo "  hparam search on layers ${HSEARCH_LAYERS[*]}..."
        PIDS=(); J=0
        for layer in "${HSEARCH_LAYERS[@]}"; do
            GPU=$(gpu_for $J)
            LOG="$LOG_DIR/hparam_off${TAG}_layer${layer}.log"
            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/train_combined.py" \
                --offsets $config --window "$WINDOW" --layer "$layer" \
                --seq-store "$SEQ_STORE" --offset-store "$OFFSET_STORE" \
                --ckpt-base "$CKPT_BASE" --hparam-search --n-trials "$N_TRIALS" \
                >"$LOG" 2>&1 &
            PIDS+=($!); J=$((J+1))
        done
        wait_all "${PIDS[@]}"
        "$PYTHON" - "$CKPT_BASE" "$WINDOW" "$TAG" "$SHARED_PARAMS" "${HSEARCH_LAYERS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
ckpt_base, window, tag, out = sys.argv[1:5]
layers = [int(x) for x in sys.argv[5:]]
best = None
for L in layers:
    p = Path(ckpt_base) / f"combined_win{int(window):02d}_off_{tag}_layer_{L:02d}" / "best_params.json"
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
    fi

    SHARED_FLAG=""; [[ -f "$SHARED_PARAMS" ]] && SHARED_FLAG="--shared-params $SHARED_PARAMS"

    # ── train all valid layers in parallel ──
    echo "  training layers ${MINL}..23 across [$GPU_LIST]..."
    PIDS=(); J=0
    for (( layer=MINL; layer<24; layer++ )); do
        GPU=$(gpu_for $J)
        LOG="$LOG_DIR/train_off${TAG}_layer${layer}.log"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/train_combined.py" \
            --offsets $config --window "$WINDOW" --layer "$layer" \
            --seq-store "$SEQ_STORE" --offset-store "$OFFSET_STORE" \
            --ckpt-base "$CKPT_BASE" $SHARED_FLAG $FORCE_FLAG \
            >"$LOG" 2>&1 &
        PIDS+=($!); J=$((J+1))
    done
    wait_all "${PIDS[@]}"
    echo "  config offset={$config} done."
done

# ── joint evaluation ──
if [[ $SKIP_EVAL -eq 0 ]]; then
    echo ""
    echo "Evaluating both configs on test split..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" "$SCRIPT_DIR/evaluate_combined.py" \
        --window "$WINDOW" --offsets-list "1,2,3,4" "1" \
        --seq-store "$SEQ_STORE" --offset-store "$OFFSET_STORE" \
        --ckpt-base "$CKPT_BASE" --out-dir "$RESULTS_DIR"
fi

echo "============================================================"
echo " Done."
echo "  Checkpoints: $CKPT_BASE"
echo "  Results    : $RESULTS_DIR/summary.csv + per_layer_overlap.png"
echo "  Job logs   : $LOG_DIR"
echo "============================================================"
