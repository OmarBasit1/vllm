#!/usr/bin/env bash
# Orchestration script for the expert-predictor pipeline.
#
# Parallelism strategy
# --------------------
# All 24 layer-training jobs for a given combo are launched simultaneously.
# Jobs are pinned to GPUs round-robin via CUDA_VISIBLE_DEVICES, so with 2 GPUs
# each GPU handles 12 layers at a time. Each job's stdout/stderr goes to its own
# log file in $LOG_DIR so the terminal stays readable.
#
# Steps:
#   1. (Optional) install optuna.
#   2. Build dataset from raw logs  [single process, uses GPU for gate-weight muls].
#   3. Hparam search: representative layers, full combo {1,2,3,4}, in parallel.
#   4. Train all 7 ablation combos × all valid layers, in parallel per combo.
#   5. Evaluate all checkpoints on the test split.
#
# Usage:
#   bash run_all.sh [options]
#
# Options:
#   --gpus N            Number of GPUs to use (default: auto-detect, max 2)
#   --gpu-ids "0 1"     Explicit GPU IDs to use (overrides --gpus)
#   --max-files N       Limit log files processed during dataset build (0 = all)
#   --skip-build        Skip dataset build (dataset already exists)
#   --skip-hparam-search  Skip hyperparameter search (use defaults or cached params)
#   --skip-eval         Skip final evaluation
#   --force-retrain     Retrain even if a checkpoint already exists
#   --n-trials N        Optuna trials per hparam search run (default: 50)

set -uo pipefail   # note: NOT -e so one failed layer job doesn't abort everything

PYTHON="${PYTHON:-/home/obasit/miniconda3/envs/vllm-moe/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_dataset"
CKPT_BASE="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_checkpoints"
OUT_DIR="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_results"
LOGS_DIR="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/all_layers_pre_gating_logs"
LOG_DIR="/export2/obasit/ClusterMoE/logs/logs3/qwen1.5_2.7B/expert_predictor_joblogs"

MAX_FILES=0
SKIP_BUILD=0
SKIP_HPARAM_SEARCH=0
SKIP_EVAL=0
FORCE_RETRAIN=0
N_TRIALS=50
NUM_GPUS=""       # empty = auto-detect
GPU_IDS=""        # e.g. "0 1"

# ── Parse CLI ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)              NUM_GPUS="$2";   shift 2 ;;
        --gpu-ids)           GPU_IDS="$2";    shift 2 ;;
        --max-files)         MAX_FILES="$2";  shift 2 ;;
        --skip-build)        SKIP_BUILD=1;    shift ;;
        --skip-hparam-search) SKIP_HPARAM_SEARCH=1; shift ;;
        --skip-eval)         SKIP_EVAL=1;     shift ;;
        --force-retrain)     FORCE_RETRAIN=1; shift ;;
        --n-trials)          N_TRIALS="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve GPU list ─────────────────────────────────────────────────────────
if [[ -n "$GPU_IDS" ]]; then
    # User gave explicit IDs, e.g. "0 1"
    read -ra GPUS <<< "$GPU_IDS"
elif [[ -n "$NUM_GPUS" ]]; then
    # User gave a count — pick first N GPUs
    GPUS=()
    for (( i=0; i<NUM_GPUS; i++ )); do GPUS+=("$i"); done
else
    # Auto-detect: cap at 2 (change the cap here if you have more)
    N_AVAIL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
    N_USE=$(( N_AVAIL < 2 ? N_AVAIL : 2 ))
    GPUS=()
    for (( i=0; i<N_USE; i++ )); do GPUS+=("$i"); done
fi

N_GPU=${#GPUS[@]}
GPU_LIST=$(IFS=,; echo "${GPUS[*]}")

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Expert Predictor Pipeline"
echo " Python : $PYTHON"
echo " GPUs   : [${GPU_LIST}]  (${N_GPU} GPU(s))"
echo " Data   : $DATA_DIR"
echo " Ckpts  : $CKPT_BASE"
echo " Job logs: $LOG_DIR"
echo "============================================================"

# ── Helpers ──────────────────────────────────────────────────────────────────

# wait_all PIDS... — wait for every PID and count failures
wait_all() {
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  WARNING: job PID=$pid exited with error" >&2
            failed=$(( failed + 1 ))
        fi
    done
    if [[ $failed -gt 0 ]]; then
        echo "  $failed job(s) failed — check logs in $LOG_DIR" >&2
    fi
}

# gpu_for INDEX — pick GPU round-robin from the GPUS array
gpu_for() { echo "${GPUS[$(( $1 % N_GPU ))]}"; }

# combo_tag "1 2 3" → "1_2_3"
combo_tag() { echo "${1// /_}"; }

# ── Step 0: Install optuna ───────────────────────────────────────────────────
echo ""
echo "Step 0: Checking optuna..."
if "$PYTHON" -c "import optuna" 2>/dev/null; then
    echo "  optuna already installed."
else
    echo "  Installing optuna..."
    "$PYTHON" -m pip install optuna --quiet \
        && echo "  optuna installed." \
        || echo "  WARNING: optuna install failed — will fall back to grid search."
fi

# ── Step 1: Build dataset ────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo ""
    echo "Step 1: Building dataset (single process, GPU ${GPUS[0]})..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" "$SCRIPT_DIR/dataset_builder.py" \
        --logs-dir "$LOGS_DIR" \
        --out-dir  "$DATA_DIR" \
        --max-files "$MAX_FILES"
    echo "  Dataset ready at $DATA_DIR"
else
    echo ""
    echo "Step 1: Skipping dataset build (--skip-build)."
fi

# Read number of layers from metadata (default 24)
NUM_LAYERS=24
META="$DATA_DIR/metadata.json"
if [[ -f "$META" ]]; then
    NUM_LAYERS=$("$PYTHON" -c "import json; print(json.load(open('$META'))['num_layers'])" 2>/dev/null || echo 24)
fi

# ── Step 2: Hyperparameter search ───────────────────────────────────────────
# Run in parallel: 5 representative layers, full combo {1,2,3,4}.
# Each job lands on a GPU round-robin and logs to its own file.
HPARAM_LAYERS=(4 8 12 16 20)

if [[ $SKIP_HPARAM_SEARCH -eq 0 ]]; then
    echo ""
    echo "Step 2: Hyperparameter search — ${#HPARAM_LAYERS[@]} layers in parallel across [${GPU_LIST}]..."
    PIDS=()
    JOB_IDX=0
    for layer in "${HPARAM_LAYERS[@]}"; do
        GPU=$(gpu_for $JOB_IDX)
        LOG="$LOG_DIR/hparam_layer${layer}.log"
        echo "  layer=${layer} → GPU ${GPU}  (log: $(basename "$LOG"))"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/train_expert_predictor.py" \
            --layer "$layer" \
            --offsets 1 2 3 4 \
            --data-dir "$DATA_DIR" \
            --ckpt-base "$CKPT_BASE" \
            --hparam-search \
            --n-trials "$N_TRIALS" \
            >"$LOG" 2>&1 &
        PIDS+=($!)
        JOB_IDX=$(( JOB_IDX + 1 ))
    done
    echo "  Waiting for hparam search jobs..."
    wait_all "${PIDS[@]}"
    echo "  Hyperparameter search done."
else
    echo ""
    echo "Step 2: Skipping hparam search (--skip-hparam-search)."
fi

# ── Step 3: Train all ablation configurations ────────────────────────────────
# For each combo: launch ALL valid layers simultaneously across GPUs,
# then wait before the next combo.
echo ""
echo "Step 3: Training ablation configs — all layers in parallel..."

CONFIGS=(
    "1"
    "2"
    "3"
    "4"
    "1 2"
    "1 2 3"
    "1 2 3 4"
)

FORCE_FLAG=""
[[ $FORCE_RETRAIN -eq 1 ]] && FORCE_FLAG="--force-retrain"

for config in "${CONFIGS[@]}"; do
    TAG=$(combo_tag "$config")
    # Minimum layer for this combo = max offset value
    MIN_LAYER=$(echo "$config" | tr ' ' '\n' | sort -n | tail -1)

    echo ""
    echo "  Combo {${config}} — layers ${MIN_LAYER}..$(( NUM_LAYERS - 1 )) in parallel..."

    PIDS=()
    JOB_IDX=0
    for (( layer=MIN_LAYER; layer<NUM_LAYERS; layer++ )); do
        GPU=$(gpu_for $JOB_IDX)
        LOG="$LOG_DIR/train_layer${layer}_combo${TAG}.log"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/train_expert_predictor.py" \
            --layer "$layer" \
            --offsets $config \
            --data-dir  "$DATA_DIR" \
            --ckpt-base "$CKPT_BASE" \
            $FORCE_FLAG \
            >"$LOG" 2>&1 &
        PIDS+=($!)
        JOB_IDX=$(( JOB_IDX + 1 ))
    done

    echo "  Launched ${#PIDS[@]} jobs across [${GPU_LIST}]. Waiting..."
    wait_all "${PIDS[@]}"
    echo "  Combo {${config}} done."
done

echo ""
echo "  All models trained."

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
if [[ $SKIP_EVAL -eq 0 ]]; then
    echo ""
    echo "Step 4: Evaluating all models on test split..."
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON" "$SCRIPT_DIR/evaluate_ablation.py" \
        --data-dir  "$DATA_DIR" \
        --ckpt-base "$CKPT_BASE" \
        --out-dir   "$OUT_DIR"
    echo "  Results written to $OUT_DIR"
else
    echo ""
    echo "Step 4: Skipping evaluation (--skip-eval)."
fi

echo ""
echo "============================================================"
echo " Done."
echo "  Dataset    : $DATA_DIR"
echo "  Checkpoints: $CKPT_BASE"
echo "  Job logs   : $LOG_DIR"
echo "  Results    : $OUT_DIR/ablation_summary.csv"
echo "               $OUT_DIR/heatmap_{kl,overlap,iou}.png"
echo "               $OUT_DIR/marginal_by_{combo,layer}.png"
echo "============================================================"
