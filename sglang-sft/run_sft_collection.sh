#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AMDPILOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_BASE="$SCRIPT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================"
echo "  SFT Trajectory Collection — Claude Opus 4.6 Executor"
echo "======================================================"
echo "Timestamp:  $TIMESTAMP"
echo "Amdpilot:   $AMDPILOT_DIR"
echo "Results:    $RESULTS_BASE"
echo ""

# All three agents (supervisor, executor, nudge) use Claude Opus 4.6 via the proxy.
# The executor's model_endpoint in task.yaml points directly to the proxy (OpenAI-compatible).
# The supervisor endpoint also points to the proxy. No AMDPILOT_EXECUTOR_USE_FRONTIER needed
# since both model_endpoint and supervisor_model_endpoint point to the same proxy.
export AMDPILOT_SUPERVISOR_MODEL_URL="http://localhost:8083/v1"
export AMDPILOT_SUPERVISOR_MODEL="claude-opus-4-6"

# Verify proxy is running
echo "Checking supervisor proxy..."
if ! curl -s localhost:8083/health | grep -q '"status": "ok"'; then
    echo "ERROR: Supervisor proxy not running on port 8083"
    echo "Start it with: source .env.supervisor && python3 tools/supervisor_proxy.py"
    exit 1
fi
echo "  Proxy OK (Claude Opus 4.6)"
echo ""

# ─── Issue #19935 ───────────────────────────────────────
echo "======================================================"
echo "  Building image: sglang-issue-19935:base"
echo "======================================================"
docker build -t sglang-issue-19935:base "$SCRIPT_DIR/issue-19935/"
echo ""

RESULTS_19935="$RESULTS_BASE/issue-19935-$TIMESTAMP"
mkdir -p "$RESULTS_19935"

echo "======================================================"
echo "  Running: SGLang #19935 (FP8 MLA decode fix)"
echo "  GPUs: 0,1,2,3  |  Hours: 2"
echo "  Results: $RESULTS_19935"
echo "======================================================"

cd "$AMDPILOT_DIR"
uv run amdpilot run "$SCRIPT_DIR/issue-19935/task.yaml" \
    --results-dir "$RESULTS_19935" \
    --hours 2 \
    --gpu "0,1,2,3" \
    --frontier-model \
    2>&1 | tee "$RESULTS_19935/run.log"

echo ""
echo "Issue #19935 complete. Results at: $RESULTS_19935"
echo ""

# ─── Issue #20187 ───────────────────────────────────────
echo "======================================================"
echo "  Building image: sglang-issue-20187:base"
echo "======================================================"
docker build -t sglang-issue-20187:base "$SCRIPT_DIR/issue-20187/"
echo ""

RESULTS_20187="$RESULTS_BASE/issue-20187-$TIMESTAMP"
mkdir -p "$RESULTS_20187"

echo "======================================================"
echo "  Running: SGLang #20187 (FP8 prefill + radix cache)"
echo "  GPUs: 0-7  |  Hours: 2"
echo "  Results: $RESULTS_20187"
echo "======================================================"

cd "$AMDPILOT_DIR"
uv run amdpilot run "$SCRIPT_DIR/issue-20187/task.yaml" \
    --results-dir "$RESULTS_20187" \
    --hours 2 \
    --gpu "0,1,2,3,4,5,6,7" \
    --frontier-model \
    2>&1 | tee "$RESULTS_20187/run.log"

echo ""
echo "Issue #20187 complete. Results at: $RESULTS_20187"
echo ""

# ─── Summary ────────────────────────────────────────────
echo "======================================================"
echo "  SFT Collection Complete"
echo "======================================================"
echo "Issue #19935 results: $RESULTS_19935"
echo "Issue #20187 results: $RESULTS_20187"
echo ""
echo "Trajectory data locations:"
echo "  $RESULTS_19935/agent_output/"
echo "  $RESULTS_20187/agent_output/"
