#!/bin/bash
# BF16 MoE OOB benchmark for AMD MI355X with TP=8.
# Runs sglang.bench_one_batch with a BF16 MoE model and checks for crashes.
# Immutable — do NOT modify this script.
set -uo pipefail

MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/Qwen3-Next-80B-A3B-Instruct}"

echo "=== BF16 MoE OOB Benchmark ==="
echo "Model: $MODEL_PATH"
echo "Starting benchmark..."

OUTPUT=$(/opt/venv/bin/python3 -m sglang.bench_one_batch \
    --model-path "$MODEL_PATH" \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 1024 \
    --dtype bfloat16 \
    --mem-fraction-static 0.9 \
    2>&1) || true

EXIT_CODE=${PIPESTATUS[0]:-$?}
echo "$OUTPUT"

# Check for GPU memory faults
if echo "$OUTPUT" | grep -qi "memory access fault\|page fault\|segfault\|SIGSEGV\|SIGBUS\|core dump"; then
    echo ""
    echo "ERROR: GPU memory access fault detected"
    echo "SCORE: 0.0"
    exit 0
fi

# Check for successful decode output
DECODE_SEC=$(echo "$OUTPUT" | grep -oP 'Decode\.\s+median latency:\s+\K[\d.]+' | tail -1)

if [ -n "$DECODE_SEC" ]; then
    DECODE_MS=$(/opt/venv/bin/python3 -c "print(f'{float(\"$DECODE_SEC\") * 1000:.1f}')")
    echo ""
    echo "Decode median (ms): $DECODE_MS | tp=8 batch=1 in=8192 out=1024"
    echo "SCORE: 100.0"
else
    echo ""
    echo "ERROR: Benchmark did not produce valid decode output"
    echo "SCORE: 0.0"
fi
