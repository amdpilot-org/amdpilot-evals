#!/bin/bash
# Launch retry worker with hardcoded server info from worker logs

RESULTS_DIR="/home/jinpan12/amdpilot/evals/kernelbench/phase3_results"
AMDPILOT_DIR="/home/jinpan12/amdpilot"

# Extract server IP from worker log (reliable source)
MODEL_SERVER=$(head -1 "$RESULTS_DIR/worker_0.log" | grep -oP 'Server: \K[0-9.]+')
MODEL_NAME=$(head -1 "$RESULTS_DIR/worker_0.log" | grep -oP 'Model: \K\S+')

echo "Model server: $MODEL_SERVER"
echo "Model name: $MODEL_NAME"

export NUM_WORKERS=1
export AMDPILOT_SUPERVISOR_MODEL_URL="http://localhost:8083/v1"
export AMDPILOT_SUPERVISOR_MODEL="claude-opus-4-6"

WORKER_LABEL=$1
TASK_FILE=$2
GPU_START=$3
GPU_END=$4

echo "[Retry2 Worker $WORKER_LABEL] Launching with GPUs $GPU_START-$GPU_END"
echo "[Retry2 Worker $WORKER_LABEL] Task file: $TASK_FILE ($(wc -l < "$TASK_FILE") tasks)"

bash "$RESULTS_DIR/worker.sh" 0 "$TASK_FILE" "$RESULTS_DIR" "$AMDPILOT_DIR" "$MODEL_SERVER" "$MODEL_NAME" 1 "$GPU_START" "$GPU_END"
