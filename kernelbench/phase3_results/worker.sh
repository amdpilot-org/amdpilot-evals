#!/bin/bash
WORKER_ID=$1
TASK_LIST=$2
RESULTS_DIR=$3
AMDPILOT_DIR=$4
MODEL_URL=$5
MODEL_NAME=$6
MAX_HOURS=$7
GPU_START=$8
GPU_END=$9
LOG_FILE="$RESULTS_DIR/worker_${WORKER_ID}.log"

GPU_IDS=""
for ((g=GPU_START; g<=GPU_END; g++)); do
    if [ -n "$GPU_IDS" ]; then GPU_IDS="$GPU_IDS,"; fi
    GPU_IDS="${GPU_IDS}${g}"
done

echo "[Worker $WORKER_ID] GPUs: $GPU_IDS, Server: $MODEL_URL, Model: $MODEL_NAME" | tee -a "$LOG_FILE"

TOTAL_LINES=$(wc -l < "$TASK_LIST")
LINE_NUM=0
COMPLETED=0
IMPROVED=0

while IFS= read -r task_dir; do
    LINE_NUM=$((LINE_NUM + 1))
    if [ $(( (LINE_NUM - 1) % ${NUM_WORKERS:-4} )) -ne "$WORKER_ID" ]; then
        continue
    fi

    task_name=$(basename "$task_dir")
    task_yaml="$task_dir/task.yaml"
    result_marker="$RESULTS_DIR/${task_name}.done"

    if [ -f "$result_marker" ]; then
        echo "[Worker $WORKER_ID] Skipping $task_name (already done)" | tee -a "$LOG_FILE"
        continue
    fi

    echo "[Worker $WORKER_ID] Starting $task_name (task $LINE_NUM/$TOTAL_LINES)" | tee -a "$LOG_FILE"
    START_TIME=$(date +%s)

    cd "$AMDPILOT_DIR"
    timeout 3600 \
        uv run amdpilot run "$task_yaml" \
            --gpu "$GPU_IDS" \
            --model-url "http://$MODEL_URL:30000/v1" \
            --model "$MODEL_NAME" \
            --hours "$MAX_HOURS" \
            --frontier-model \
            --results-dir "$RESULTS_DIR/$task_name" \
        >> "$LOG_FILE" 2>&1 || true

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "[Worker $WORKER_ID] Finished $task_name in ${ELAPSED}s" | tee -a "$LOG_FILE"

    touch "$result_marker"
    COMPLETED=$((COMPLETED + 1))

    SCORE_FILE="$RESULTS_DIR/$task_name/scoreboard.jsonl"
    if [ -f "$SCORE_FILE" ]; then
        BEST_SCORE=$(tail -1 "$SCORE_FILE" 2>/dev/null | grep -oP '"metric_value":\s*[\d.]+' | grep -oP '[\d.]+' || echo "0")
        if [ "$(echo "$BEST_SCORE > 50" | bc 2>/dev/null)" = "1" ]; then
            IMPROVED=$((IMPROVED + 1))
            echo "[Worker $WORKER_ID] IMPROVED: $task_name scored $BEST_SCORE" | tee -a "$LOG_FILE"
        fi
    fi

done < "$TASK_LIST"

echo "[Worker $WORKER_ID] DONE. Completed: $COMPLETED, Improved: $IMPROVED" | tee -a "$LOG_FILE"
