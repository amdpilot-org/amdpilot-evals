#!/bin/bash
set -euo pipefail

# Auto-detect the host network interface for Gloo/NCCL sockets if the
# bench_config.env doesn't already pin one. Without GLOO_SOCKET_IFNAME
# the Megatron distributed init fails fast (3-5s) before training
# starts and Phase 1 / executor trials can't get a real metric.
if [ -f /workspace/bench_config.env ]; then
    echo '>>> Loading bench_config.env overrides'
    source /workspace/bench_config.env
fi
if [ -z "${GLOO_SOCKET_IFNAME:-}" ] && [ -x /workspace/detect_interface.sh ]; then
    echo '>>> Auto-detecting GLOO_SOCKET_IFNAME via /workspace/detect_interface.sh'
    eval "$(/workspace/detect_interface.sh)"
fi
echo ">>> GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-<unset>} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}"

cd /workspace/primus_train/Primus

echo '=== Starting Qwen3-30B-A3B MFU Benchmark ==='
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

TRAIN_ITERS="${TRAIN_ITERS:-10}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-8192}"
EP_SIZE="${EP_SIZE:-8}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-5}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-block}"
TURBO_DEEPEP_NUM_CU="${TURBO_DEEPEP_NUM_CU:-80}"
TURBO_SYNC_FREE_MOE_STAGE="${TURBO_SYNC_FREE_MOE_STAGE:-1}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

echo ">>> Config: mbs=${MICRO_BATCH_SIZE}, gbs=${GLOBAL_BATCH_SIZE}, seq=${SEQ_LENGTH}, EP=${EP_SIZE}"
echo ">>> Recompute: ${RECOMPUTE_NUM_LAYERS} layers, ${RECOMPUTE_GRANULARITY}/${RECOMPUTE_METHOD}"
echo ">>> Train iters: ${TRAIN_ITERS}"

OUTPUT=$(./primus-cli direct \
  -- train pretrain --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml \
  --train_iters ${TRAIN_ITERS} \
  --micro_batch_size ${MICRO_BATCH_SIZE} \
  --global_batch_size ${GLOBAL_BATCH_SIZE} \
  --seq_length ${SEQ_LENGTH} \
  --max_position_embeddings ${MAX_POSITION_EMBEDDINGS} \
  --expert_model_parallel_size ${EP_SIZE} \
  --mock_data True \
  --disable_last_saving True \
  --moe_use_legacy_grouped_gemm True \
  --use_turbo_grouped_mlp True \
  --use_turbo_attention True \
  --enable_primus_turbo True \
  --use_turbo_deepep True \
  --turbo_deepep_num_cu ${TURBO_DEEPEP_NUM_CU} \
  --turbo_sync_free_moe_stage ${TURBO_SYNC_FREE_MOE_STAGE} \
  --enable_experimental True \
  --apply_rope_fusion True \
  --cross_entropy_fusion_impl te \
  --cross_entropy_loss_fusion True \
  --use_precision_aware_optimizer True \
  --main_grads_dtype bf16 \
  --exp_avg_dtype bf16 \
  --exp_avg_sq_dtype bf16 \
  --recompute_num_layers ${RECOMPUTE_NUM_LAYERS} \
  --recompute_granularity ${RECOMPUTE_GRANULARITY} \
  --recompute_method ${RECOMPUTE_METHOD} \
  --disable_wandb True \
  --disable_tensorboard True \
  ${EXTRA_FLAGS} 2>&1) || true

echo "$OUTPUT"
echo "$OUTPUT" > /workspace/bench_output.log

# Extract per-iteration instant TFLOP/s/GPU from the Megatron log.
# Format: "throughput per GPU (TFLOP/s/GPU): <instant>/<running_avg>"
# We extract <instant> (the number before the slash).
TFLOPS_VALUES=$(echo "$OUTPUT" | grep -oP 'throughput per GPU \(TFLOP/s/GPU\):\s*\K[\d.]+' || true)

if [ -z "$TFLOPS_VALUES" ]; then
    # Fallback: try "tflops/gpu:" pattern
    TFLOPS_VALUES=$(echo "$OUTPUT" | grep -oP 'tflops/gpu:\s*\K[\d.]+' || true)
fi

if [ -z "$TFLOPS_VALUES" ]; then
    TFLOPS_VALUES=$(echo "$OUTPUT" | grep -oP 'TFLOP/s/GPU[):\s]*\K[\d.]+' || true)
fi

if [ -z "$TFLOPS_VALUES" ]; then
    echo 'ERROR: Could not extract TFLOP/s/GPU from output'
    echo 'TFLOPS_PER_GPU: 0'
    echo 'METRIC: 0'
    exit 1
fi

readarray -t VALUES <<< "$TFLOPS_VALUES"
TOTAL_ITERS=${#VALUES[@]}
echo "=== Found $TOTAL_ITERS iteration TFLOP/s values ==="

SKIP=2
SUM=0
COUNT=0
for i in "${!VALUES[@]}"; do
    val="${VALUES[$i]}"
    echo "  Iter $((i+1)): ${val} TFLOP/s/GPU"
    if [ "$i" -ge "$SKIP" ]; then
        SUM=$(python3 -c "print($SUM + $val)")
        COUNT=$((COUNT + 1))
    fi
done

if [ "$COUNT" -eq 0 ]; then
    echo 'ERROR: Not enough iterations after skipping warmup'
    echo 'TFLOPS_PER_GPU: 0'
    echo 'METRIC: 0'
    exit 1
fi

AVG=$(python3 -c "print(round($SUM / $COUNT, 2))")
echo ''
echo "=== Steady-state average (iters $((SKIP+1))-${TOTAL_ITERS}): ${AVG} TFLOP/s/GPU ==="
# Emit both the domain-specific label and the generic METRIC label so
# existing task.yaml regexes and newer canonical wrappers both work.
echo "TFLOPS_PER_GPU: ${AVG}"
echo "METRIC: ${AVG}"
