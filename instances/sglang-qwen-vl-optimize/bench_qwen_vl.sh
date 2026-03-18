#!/bin/bash
# Qwen3-VL-8B serving throughput benchmark for AMD MI355X.
# Self-contained: starts server, runs warmup + benchmark, reports metric.
# Agent can override settings via /workspace/bench_config.env.
set -eo pipefail

if [ -f /workspace/bench_config.env ]; then
    source /workspace/bench_config.env
fi

PORT="${BENCH_PORT:-30000}"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

export SGLANG_DISABLE_CUDNN_CHECK=1

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Kill leftover sglang server processes (avoid pkill -f which kills the shell)
ps aux | grep -E "sglang\.(launch_server|serve)" | grep -v grep \
    | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# Start sglang server
/opt/venv/bin/python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --attention-backend "$ATTENTION_BACKEND" \
    $EXTRA_SERVER_ARGS > /tmp/sglang_server.log 2>&1 &
SERVER_PID=$!

echo "Starting sglang server (PID: $SERVER_PID, attention: $ATTENTION_BACKEND)..."
READY=0
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready after $((i * 5))s"
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died during startup. Log tail:"
        tail -50 /tmp/sglang_server.log
        exit 1
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Server did not become ready within 600s. Log tail:"
    tail -50 /tmp/sglang_server.log
    exit 1
fi

BENCH_ARGS="--backend sglang --model $MODEL --port $PORT \
    --dataset-name image --num-prompts 128 \
    --random-input-len 4000 --random-output-len 2000 \
    --random-range-ratio 1.0 --image-count 1 \
    --image-resolution 720p --image-content random \
    --max-concurrency 16 --seed 123 --warmup-requests 0"

echo ""
echo "=== Warmup run ==="
/opt/venv/bin/python3 -m sglang.bench_serving $BENCH_ARGS 2>&1 || true

echo ""
echo "=== Benchmark run ==="
OUTPUT=$(/opt/venv/bin/python3 -m sglang.bench_serving $BENCH_ARGS 2>&1) || true

echo "$OUTPUT"

THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Output token throughput \(tok/s\):\s+\K[\d.]+' | tail -1)

if [ -n "$THROUGHPUT" ]; then
    echo ""
    echo "Output throughput (tok/s): $THROUGHPUT | concurrency=16 model=Qwen3-VL-8B"
else
    echo "ERROR: Could not extract output throughput from benchmark output" >&2
    exit 1
fi
