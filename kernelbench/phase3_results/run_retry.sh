#!/bin/bash
RESULTS_DIR="/home/jinpan12/amdpilot/evals/kernelbench/phase3_results"
WORKER_LABEL=$1
TASK_FILE=$2
GPU_START=$3
GPU_END=$4

echo "[Retry $WORKER_LABEL] Starting retry run"
bash "$RESULTS_DIR/launch_offload.sh" "$WORKER_LABEL" "$TASK_FILE" "$GPU_START" "$GPU_END"
echo "[Retry $WORKER_LABEL] Done"
