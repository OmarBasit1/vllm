#!/usr/bin/env bash
# UVA vs Prefetch — max batch size (max-num-seqs) sweep.
#
# Question: is there a concurrency point where speculative Prefetch beats
# on-demand UVA? (n=100 capacity probes at max-concurrency=16 showed UVA
# ~2.2x faster; hypothesis is prefetch may win at low concurrency where
# PCIe has slack to hide transfer behind compute, per exp1 finding that
# chunked-vs-nonchunked barely matters, so prefill mode is fixed here to
# non-chunked to isolate the batch-size axis).
#
# GPU memory budget is IDENTICAL across every run: --gpu-memory-utilization
# and --cpu-offload-gb are fixed in COMMON regardless of batch size — vLLM's
# KV-cache pool is sized from a profiling pass keyed on gpu-memory-utilization
# and max-num-batched-tokens (both fixed), not on max-num-seqs, so raising the
# batch cap does not change how much GPU memory is reserved.
#
# Usage:
#   ./run_batch_sweep.sh sweep [cfg...]     # BATCH_GRID sweep (default below)
#   BATCH_GRID="1 4 16" ./run_batch_sweep.sh sweep uva
set -uo pipefail

ENV_BIN=${ENV_BIN:-/home/obasit/miniconda3/envs/vllm-moe-expert-cache/bin}
EXP=${EXP:-/export3/obasit/ClusterMoE/experiments/exp3_batch_sweep}
DATASET=/export3/obasit/ClusterMoE/experiments/exp1_chunk_interference/dataset/wildchat_1k.jsonl
SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
PORT=8000
NPROMPTS=${NPROMPTS:-50}
BATCH_GRID=${BATCH_GRID:-"1 2 4 8 16 24 32"}
mkdir -p "$EXP/results" "$EXP/logs" "$EXP/monitor"

export HF_HOME=/export3/obasit/hf_home
export LD_LIBRARY_PATH=$ENV_BIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
# expert_cache eagerly loads every expert and CPU-pins it at startup, and the
# HF weights stream from /export3, so engine bring-up can exceed the default
# ready timeout. Raise it so a slow-but-healthy startup isn't killed (the
# 3-attempt retry in start_server_ready then only fires on genuine failures).
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}

CURL="env -u LD_LIBRARY_PATH curl"
NUMACTL="$ENV_BIN/numactl --cpunodebind=1 --localalloc"

# Fixed across every run: model, GPU memory budget, CPU-offload budget,
# max-model-len, prefill mode (non-chunked). Only --max-num-seqs varies.
COMMON="$MODEL --max-model-len 8192 --gpu-memory-utilization 0.87 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-num-batched-tokens 8192 --port $PORT \
  --compilation-config {\"fast_moe_cold_start\":false}"
# --enforce-eager on every arm (not just expert_cache): expert_cache's
# cache-miss resolution is inherently data-dependent (inspects actual
# topk_ids values at runtime), which is fundamentally incompatible with
# CUDA graph capture, so it always runs eager. UVA and Prefetch normally run
# with CUDA graphs enabled; forcing eager here too is required for a fair
# apples-to-apples 3-way comparison rather than comparing expert_cache's
# eager numbers against the other two backends' graph-mode numbers.
UVA="--offload-backend uva --cpu-offload-gb 27 --cpu-offload-params w13_weight w2_weight \
  --enforce-eager"
PREFETCH="--offload-backend prefetch --offload-group-size 2 --offload-num-in-group 1 \
  --offload-prefetch-step 1 --offload-params w13_weight w2_weight --enforce-eager"
# expert_cache is now a single GLOBAL cross-layer expert cache: one GPU pool of
# --expert-cache-capacity experts is shared across ALL 48 layers (not a per-layer
# hot set). Experts a step needs that are resident ("hits") cost no transfer; the
# rest ("misses") are loaded on demand, evicting the least-valuable resident
# experts (graded LFU). The pool is sized for parity with the UVA arm's
# --cpu-offload-gb 27: this model has 128 experts/layer at ~9.0 MiB/expert (bf16
# w13+w2), so 3072 pooled experts ~= 3072*9.0 MiB ~= 27 GiB (plus a ~1.15 GiB
# contiguous per-wave compute buffer). --expert-cache-budget-gb 27 fails fast if
# the *measured* resident footprint drifts >15% from the UVA bar.
#
# --expert-cache-waves selects the per-layer schedule: 1 = stall (load all misses,
# then one kernel over hits+misses); 2 = overlap (compute resident hits while the
# misses stream in, then compute the misses; the runner sums the two). We sweep
# BOTH as separate arms for a head-to-head. Predicted upcoming-layer experts are
# prefetched within a measured PCIe budget (--expert-cache-predict-k /
# --expert-cache-prefetch-horizon). --expert-cache-max-transient-experts is gone
# (the old per-wave scratch no longer exists).
EXPERT_CACHE_COMMON="--offload-backend expert_cache --expert-cache-capacity 3072 \
  --expert-cache-predict-k 16 --expert-cache-prefetch-horizon 1 \
  --expert-cache-budget-gb 27 --expert-cache-params w13_weight w2_weight \
  --enforce-eager"
EXPERT_CACHE_W1="$EXPERT_CACHE_COMMON --expert-cache-waves 1"
EXPERT_CACHE_W2="$EXPERT_CACHE_COMMON --expert-cache-waves 2"

declare -A BACKEND_ARGS=(
  [uva]="$UVA" [prefetch]="$PREFETCH"
  [expert_cache_w1]="$EXPERT_CACHE_W1" [expert_cache_w2]="$EXPERT_CACHE_W2"
)
ALL_CFGS="uva prefetch expert_cache_w1 expert_cache_w2"

SERVER_PID=""

start_server() {  # $1=cfg $2=batch $3=stage
  local cfg=$1 batch=$2 stage=$3
  echo "[$(date +%T)] starting server: $cfg batch=$batch"
  # shellcheck disable=SC2086
  $NUMACTL $ENV_BIN/vllm serve $COMMON ${BACKEND_ARGS[$cfg]} --max-num-seqs "$batch" \
    > "$EXP/logs/server_${cfg}_b${batch}_${stage}.log" 2>&1 &
  SERVER_PID=$!
}

wait_ready() {
  for _ in $(seq 1 240); do
    $CURL -sf "localhost:$PORT/health" > /dev/null && return 0
    kill -0 "$SERVER_PID" 2> /dev/null || { echo "server died (see log)"; return 1; }
    sleep 10
  done
  echo "server readiness timeout"; return 1
}

start_server_ready() {  # $1=cfg $2=batch $3=stage — retries: UVA pin_memory flakily fails at startup
  local cfg=$1 batch=$2 stage=$3
  for attempt in 1 2 3; do
    start_server "$cfg" "$batch" "$stage"
    wait_ready && return 0
    echo "[$(date +%T)] startup attempt $attempt failed for $cfg batch=$batch, retrying"
    stop_server
  done
  echo "[$(date +%T)] giving up on $cfg batch=$batch after 3 attempts"
  return 1
}

stop_server() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2> /dev/null
  wait "$SERVER_PID" 2> /dev/null
  SERVER_PID=""
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 1000 ]; do
    sleep 5
  done
}

run_bench() {  # $1=cfg $2=batch
  local cfg=$1 batch=$2 tag="b${batch}"
  bash "$SCRIPTDIR/monitor.sh" "$cfg" "$tag" "$EXP/monitor" "$PORT" &
  local mon_pid=$!
  $NUMACTL $ENV_BIN/vllm bench serve \
    --backend openai --host 127.0.0.1 --port $PORT --model $MODEL \
    --dataset-name custom --dataset-path "$DATASET" \
    --custom-output-len -1 --num-prompts "$NPROMPTS" --seed 0 --disable-shuffle \
    --ignore-eos --num-warmups 3 --burstiness 1.0 \
    --request-rate inf --max-concurrency "$batch" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir "$EXP/results" \
    --result-filename "${cfg}_${tag}.json" \
    --metadata "backend=$cfg" "max_num_seqs=$batch" \
    >> "$EXP/logs/bench_${cfg}_${tag}.log" 2>&1
  local rc=$?
  kill "$mon_pid" 2> /dev/null; pkill -P "$mon_pid" 2> /dev/null
  echo "[$(date +%T)] bench $cfg/$tag done (rc=$rc)"
}

check_offloader() {  # $1=cfg $2=batch $3=stage
  local log="$EXP/logs/server_${1}_b${2}_${3}.log"
  grep -m1 "Offloader" "$log" || echo "WARNING: no 'Offloader' line in $log"
}

cmd=${1:-sweep}; shift || true
cfgs=${*:-$ALL_CFGS}

case $cmd in
  sweep)
    for cfg in $cfgs; do
      for batch in $BATCH_GRID; do
        start_server_ready "$cfg" "$batch" sweep || continue
        check_offloader "$cfg" "$batch" sweep
        run_bench "$cfg" "$batch"
        stop_server
      done
    done
    ;;
  *) echo "unknown command: $cmd"; exit 1 ;;
esac
echo "[$(date +%T)] all done: $cmd"
