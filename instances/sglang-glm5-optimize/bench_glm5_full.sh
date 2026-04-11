#!/bin/bash
# Full-model benchmark for GLM-5 on 4x AMD MI355X (TP=4).
# Used by the root node to validate kernel optimizations from proxy children.
set -eo pipefail

if [ -f /workspace/bench_config.env ]; then
    source /workspace/bench_config.env
fi

PORT="${BENCH_PORT:-30000}"
MODEL="zai-org/GLM-5"
TP="${BENCH_TP:-4}"

export SGLANG_DISABLE_CUDNN_CHECK=1

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    ps aux | grep -E "sglang::|sglang\.(launch_server|serve)" | grep -v grep \
        | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    fuser -k "${PORT}/tcp" 2>/dev/null || true
}
trap cleanup EXIT

ps aux | grep -E "sglang::|sglang\.(launch_server|serve)" | grep -v grep \
    | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 3

echo "Starting GLM-5 FULL model server (tp=$TP, real weights, port=$PORT)..."

/opt/venv/bin/python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --tp "$TP" \
    --mem-fraction-static 0.85 \
    ${EXTRA_SERVER_ARGS:-} > /tmp/sglang_server.log 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready after $((i * 5))s"
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died during startup. Log tail:"
        tail -80 /tmp/sglang_server.log
        exit 1
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Server did not become ready within 600s. Log tail:"
    tail -80 /tmp/sglang_server.log
    exit 1
fi

echo ""
echo "=== Warmup ==="
/opt/venv/bin/python3 -m sglang.bench_serving \
    --backend sglang --model "$MODEL" --port "$PORT" \
    --dataset-name random --num-prompts 4 \
    --random-input-len 1024 --random-output-len 512 \
    --random-range-ratio 1.0 \
    --max-concurrency 4 --seed 42 --warmup-requests 0 2>&1 || true

echo ""
echo "=== Benchmark run (tp=$TP, full model) ==="
OUTPUT=$(/opt/venv/bin/python3 -m sglang.bench_serving \
    --backend sglang --model "$MODEL" --port "$PORT" \
    --dataset-name random --num-prompts 32 \
    --random-input-len 1024 --random-output-len 512 \
    --random-range-ratio 1.0 \
    --max-concurrency 16 --seed 123 --warmup-requests 0 2>&1) || true

echo "$OUTPUT"

THROUGHPUT=$(echo "$OUTPUT" | grep -oP 'Output (token )?throughput \(tok/s\):\s+\K[\d.]+' | tail -1)

if [ -n "$THROUGHPUT" ]; then
    echo ""
    echo "Output throughput (tok/s): $THROUGHPUT | tp=$TP full model=GLM-5"
else
    echo "ERROR: Could not extract throughput from benchmark output" >&2
    exit 1
fi
