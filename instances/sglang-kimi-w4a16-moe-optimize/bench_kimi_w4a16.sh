#!/bin/bash
# Kimi-K2.5 W4A16 decode latency benchmark for AMD MI355X with TP=8.
# Immutable parameters — do NOT change model, tp, batch, input/output lengths.
#
# The agent may write /workspace/bench_config.env to set environment variables
# (e.g. backend selection, attention backend flags). This file is sourced if
# present, ensuring the verification run uses the same configuration.
set -euo pipefail

# Defaults — overridable via bench_config.env
export SGLANG_ROCM_FUSED_DECODE_MLA="${SGLANG_ROCM_FUSED_DECODE_MLA:-0}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-triton}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-aiter}"

# Source agent-provided config if it exists
if [ -f /workspace/bench_config.env ]; then
    source /workspace/bench_config.env
fi

OUTPUT=$(/opt/venv/bin/python3 -m sglang.bench_one_batch \
    --model-path /root/.cache/huggingface/Kimi-K2.5 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 2048 \
    --dtype bfloat16 \
    --decode-attention-backend "$DECODE_ATTENTION_BACKEND" \
    --prefill-attention-backend "$PREFILL_ATTENTION_BACKEND" \
    --mem-fraction-static 0.9 \
    2>&1) || true

echo "$OUTPUT"

DECODE_SEC=$(echo "$OUTPUT" | grep -oP 'Decode\.\s+median latency:\s+\K[\d.]+' | tail -1)

if [ -n "$DECODE_SEC" ]; then
    DECODE_MS=$(/opt/venv/bin/python3 -c "print(f'{float(\"$DECODE_SEC\") * 1000:.1f}')")
    echo "Decode median (ms): $DECODE_MS | tp=8 batch=1 in=8192 out=2048 decode=$DECODE_ATTENTION_BACKEND"
else
    echo "ERROR: Could not extract decode median from benchmark output" >&2
    exit 1
fi
