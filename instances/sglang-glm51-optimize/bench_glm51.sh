#!/bin/bash
# GLM-5.1-FP8 decode latency benchmark for AMD MI355X with TP=8.
# REALISTIC settings: 8192 input, 2048 output (matching kimi-k2.5-optimize).
# Immutable parameters — do NOT change model, tp, batch, input/output lengths.
#
# The agent may write /workspace/bench_config.env to set environment variables
# (e.g. backend selection). This file is sourced if present, ensuring the
# verification run uses the same configuration as the agent's run.
set -euo pipefail

# Defaults — overridable via bench_config.env
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-}"

# Source agent-provided config if it exists (env vars for backend selection, etc.)
if [ -f /workspace/bench_config.env ]; then
    source /workspace/bench_config.env
fi

# Build optional backend flags
EXTRA_FLAGS=""
if [ -n "$DECODE_ATTENTION_BACKEND" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --decode-attention-backend $DECODE_ATTENTION_BACKEND"
fi
if [ -n "$PREFILL_ATTENTION_BACKEND" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --prefill-attention-backend $PREFILL_ATTENTION_BACKEND"
fi

OUTPUT=$(/opt/venv/bin/python3 -m sglang.bench_one_batch \
    --model-path zai-org/GLM-5.1-FP8 \
    --tensor-parallel-size 8 \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 2048 \
    --dtype bfloat16 \
    --quantization fp8 \
    --mem-fraction-static 0.9 \
    $EXTRA_FLAGS \
    2>&1) || true

echo "$OUTPUT"

DECODE_SEC=$(echo "$OUTPUT" | grep -oP 'Decode\.\s+median latency:\s+\K[\d.]+' | tail -1)

if [ -n "$DECODE_SEC" ]; then
    DECODE_MS=$(/opt/venv/bin/python3 -c "print(f'{float(\"$DECODE_SEC\") * 1000:.1f}')")
    echo "Decode median (ms): $DECODE_MS | tp=8 batch=1 in=8192 out=2048"
else
    echo "ERROR: Could not extract decode median from benchmark output" >&2
    exit 1
fi
