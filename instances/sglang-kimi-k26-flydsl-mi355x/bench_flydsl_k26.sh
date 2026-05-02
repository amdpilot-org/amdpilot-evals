#!/bin/bash
#
# Kimi-K2.6 MI355X FlyDSL-optimization benchmark.
#
# PRIMARY metric: output_throughput_tok_s at concurrency=40
#   (sglang.bench_serving, random-input=10240, random-output=512,
#    num-prompts=160). This is where FlyDSL shines in the blog
#    (https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)
#    — decode-dominated, MoE-bound regime.
#
# GUARD metric: decode_bs1_in8k (sglang.bench_one_batch_server at
#   batch=1 input=8192 output=1024). This reproduces the PR #23381
#   measurement column (38.05 tok/s on the published baseline) so
#   any FlyDSL configuration that regresses the BS=1 path is flagged.
#
# Server launch flags are PR #23381 canonical. Immutables per
# baseline_contract.required_flags in task.yaml:
#   --tensor-parallel-size 4
#   --decode-attention-backend triton
#   --prefill-attention-backend aiter
#   --mem-fraction-static 0.85
#   --context-length 128000
#   --disable-custom-all-reduce
#
# The agent configures trial knobs by writing /workspace/bench_config.env:
#   export AITER_USE_FLYDSL_MOE=1
#   export AITER_USE_FLYDSL_MOE_STAGE1=1
#   export AITER_USE_FLYDSL_MOE_STAGE2=1
#   export FLYDSL_W4A16_HYBRID=w2_bf16
#   export AITER_ENFORCE_DSL=1
#   export EXTRA_SERVER_FLAGS="--enable-torch-compile --disable-radix-cache"

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/sgl-workspace/models/Kimi-K2.6}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
HEALTH_URL="http://${HOST}:${PORT}/health"
SERVER_LOG="/tmp/sglang_server.log"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"   # 15 min — CUDA graph capture

# ---- Load agent-supplied overrides (trial knobs) ----
if [ -f /workspace/bench_config.env ]; then
    echo ">>> Loading /workspace/bench_config.env"
    set -a
    # shellcheck disable=SC1091
    source /workspace/bench_config.env
    set +a
fi

EXTRA_SERVER_FLAGS="${EXTRA_SERVER_FLAGS:-}"

# ---- Decide whether to reuse an existing server or launch a fresh one ----
LAUNCHED_BY_US=0
if curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
    echo ">>> Re-using pre-existing SGLang server on :${PORT}"
else
    echo ">>> Launching SGLang server (K2.6, TP=4)"
    echo ">>> EXTRA_SERVER_FLAGS=${EXTRA_SERVER_FLAGS}"
    : > "$SERVER_LOG"
    # Canonical PR #23381 launch. Do NOT modify these flags; the
    # agent extends via EXTRA_SERVER_FLAGS (e.g. --enable-torch-compile,
    # --disable-radix-cache) only.
    /opt/venv/bin/python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --served-model-name Kimi-K2.6 \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        --decode-attention-backend triton \
        --prefill-attention-backend aiter \
        --disable-custom-all-reduce \
        --mem-fraction-static 0.85 \
        --context-length 128000 \
        --skip-server-warmup \
        --reasoning-parser kimi_k2 \
        --tool-call-parser kimi_k2 \
        --watchdog-timeout 1200 \
        --host "$HOST" --port "$PORT" \
        ${EXTRA_SERVER_FLAGS} \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    LAUNCHED_BY_US=1
    echo ">>> server PID=${SERVER_PID}, waiting up to ${STARTUP_TIMEOUT_S}s for /health"

    SLEEP_S=5
    DEADLINE=$(( $(date +%s) + STARTUP_TIMEOUT_S ))
    while true; do
        if curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
            echo ">>> /health OK"
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server PID ${SERVER_PID} exited during startup; last 80 log lines:" >&2
            tail -80 "$SERVER_LOG" >&2 || true
            exit 1
        fi
        if [ "$(date +%s)" -ge "$DEADLINE" ]; then
            echo "ERROR: /health did not respond within ${STARTUP_TIMEOUT_S}s; last 80 log lines:" >&2
            tail -80 "$SERVER_LOG" >&2 || true
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
        sleep "$SLEEP_S"
    done
fi

# ---- Cleanup on exit (only kill the server we started) ----
cleanup() {
    if [ "$LAUNCHED_BY_US" = "1" ] && [ -n "${SERVER_PID:-}" ]; then
        # Polite TERM, then KILL after 10s. Never use pkill -f; it can
        # match the amdpilot runtime shell and exit 137 the trial.
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---- PRIMARY: concurrency=40 bench_serving ----
echo ">>> PRIMARY: sglang.bench_serving concurrency=40 random-input=10240 random-output=512 num-prompts=160"
OUT40=$(/opt/venv/bin/python3 -m sglang.bench_serving \
    --model Kimi-K2.6 \
    --dataset-name random \
    --random-input 10240 \
    --random-output 512 \
    --num-prompts 160 \
    --max-concurrency 40 \
    --request-rate inf \
    --random-range-ratio 1.0 \
    --host "$HOST" --port "$PORT" 2>&1) || {
    echo "ERROR: bench_serving failed:" >&2
    echo "$OUT40" >&2
    exit 1
}
echo "$OUT40"
# Blog+SGLang line format: "Output token throughput (tok/s):   <float>"
THROUGHPUT=$(echo "$OUT40" | grep -oP 'Output token throughput \(tok/s\):\s+\K[\d.]+' | tail -1 || true)
[ -z "${THROUGHPUT:-}" ] && THROUGHPUT="0"

# ---- GUARD: BS=1 bench_one_batch_server at input=8192 ----
echo ">>> GUARD: sglang.bench_one_batch_server batch=1 input=8192 output=1024"
OUT1=$(/opt/venv/bin/python3 -m sglang.bench_one_batch_server \
    --model-path "$MODEL_PATH" \
    --base-url "http://${HOST}:${PORT}" \
    --dataset-name random \
    --batch-size 1 \
    --input-len 8192 \
    --output-len 1024 \
    --skip-warmup 2>&1) || {
    echo "WARN: bench_one_batch_server failed; decode_bs1_in8k=0 will trigger guard" >&2
    echo "$OUT1" >&2
}
echo "$OUT1"
# bench_one_batch_server emits JSON lines; grab "output_throughput": <float>
DECODE_BS1=$(echo "$OUT1" | grep -oP '"output_throughput":\s*\K[\d.]+' | tail -1 || true)
if [ -z "${DECODE_BS1:-}" ]; then
    # Fallback pattern in case the field is renamed in newer SGLang
    DECODE_BS1=$(echo "$OUT1" | grep -oP 'Output throughput \(tok/s\):\s+\K[\d.]+' | tail -1 || true)
fi
[ -z "${DECODE_BS1:-}" ] && DECODE_BS1="0"

echo ""
echo "=============================="
echo "output_throughput_tok_s: ${THROUGHPUT} | concurrency=40 in=10240 out=512 decode_bs1_in8k=${DECODE_BS1}"
