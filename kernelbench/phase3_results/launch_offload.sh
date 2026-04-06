#!/bin/bash
# Launch offload workers to redistribute remaining tasks

RESULTS_DIR="/home/jinpan12/amdpilot/evals/kernelbench/phase3_results"
AMDPILOT_DIR="/home/jinpan12/amdpilot"

# Get model server from worker 0's running process
MODEL_SERVER=$(ps aux | grep "worker.sh 0" | grep -v grep | awk '{print $17}')
MODEL_NAME=$(ps aux | grep "worker.sh 0" | grep -v grep | awk '{print $18}')

echo "Model server: $MODEL_SERVER"
echo "Model name: $MODEL_NAME"

export NUM_WORKERS=1
export AMDPILOT_SUPERVISOR_MODEL_URL="http://localhost:8083/v1"
export AMDPILOT_SUPERVISOR_MODEL="claude-opus-4-6"

WORKER_LABEL=$1
TASK_FILE=$2
GPU_START=$3
GPU_END=$4

echo "[Offload Worker $WORKER_LABEL] Launching with GPUs $GPU_START-$GPU_END"
echo "[Offload Worker $WORKER_LABEL] Task file: $TASK_FILE ($(wc -l < "$TASK_FILE") tasks)"

bash "$RESULTS_DIR/worker.sh" 0 "$TASK_FILE" "$RESULTS_DIR" "$AMDPILOT_DIR" "$MODEL_SERVER" "$MODEL_NAME" 1 "$GPU_START" "$GPU_END"
