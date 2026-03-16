#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMDPILOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTANCES_DIR="$SCRIPT_DIR/instances"
RESULTS_DIR="$SCRIPT_DIR/phase3_results"
MODEL_URL="${KERNELBENCH_SERVER_ADDRESS:-10.235.27.218}"
SESSION_NAME="kernelbench-phase3"
MAX_HOURS="${MAX_HOURS_PER_PROBLEM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Frontier model (Claude Opus 4.6) for Supervisor and Nudge agents
SUPERVISOR_PROXY_URL="http://localhost:8083/v1"
export AMDPILOT_SUPERVISOR_MODEL_URL="$SUPERVISOR_PROXY_URL"
export AMDPILOT_SUPERVISOR_MODEL="claude-opus-4-6"

mkdir -p "$RESULTS_DIR"

if [ ! -d "$INSTANCES_DIR" ]; then
    echo "No instances found. Run generate_tasks.py first."
    exit 1
fi

TASKS=($(ls -d "$INSTANCES_DIR"/kernelbench-L*/ 2>/dev/null | sort))
TOTAL=${#TASKS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No task instances found in $INSTANCES_DIR"
    exit 1
fi

echo "======================================================"
echo "KernelBench Phase 3 - Full amdpilot Pipeline"
echo "======================================================"
echo "Total tasks:   $TOTAL"
echo "Workers:       $NUM_WORKERS"
echo "Hours/problem: $MAX_HOURS"
echo "Executor:      Qwen3.5-397B-A17B @ $MODEL_URL:30000"
echo "Supervisor:    Claude Opus 4.6 @ $SUPERVISOR_PROXY_URL"
echo "Nudge:         Claude Opus 4.6 @ $SUPERVISOR_PROXY_URL"
echo "Results dir:   $RESULTS_DIR"
echo "======================================================"

# Distribute GPUs across workers
GPUS_PER_WORKER=$((8 / NUM_WORKERS))
if [ "$GPUS_PER_WORKER" -lt 1 ]; then
    GPUS_PER_WORKER=1
fi

# Create the task list file
TASK_LIST="$RESULTS_DIR/task_list.txt"
> "$TASK_LIST"
for task_dir in "${TASKS[@]}"; do
    task_name=$(basename "$task_dir")
    echo "$task_dir" >> "$TASK_LIST"
done

# Create worker script
WORKER_SCRIPT="$RESULTS_DIR/worker.sh"
cat > "$WORKER_SCRIPT" << 'WORKEREOF'
#!/bin/bash
WORKER_ID=$1
TASK_LIST=$2
RESULTS_DIR=$3
AMDPILOT_DIR=$4
MODEL_URL=$5
MAX_HOURS=$6
GPU_START=$7
GPU_END=$8
LOG_FILE="$RESULTS_DIR/worker_${WORKER_ID}.log"

GPU_IDS=""
for ((g=GPU_START; g<=GPU_END; g++)); do
    if [ -n "$GPU_IDS" ]; then GPU_IDS="$GPU_IDS,"; fi
    GPU_IDS="${GPU_IDS}${g}"
done

echo "[Worker $WORKER_ID] GPUs: $GPU_IDS, Log: $LOG_FILE" | tee -a "$LOG_FILE"

TOTAL_LINES=$(wc -l < "$TASK_LIST")
LINE_NUM=0
COMPLETED=0
IMPROVED=0

while IFS= read -r task_dir; do
    LINE_NUM=$((LINE_NUM + 1))
    # Round-robin assignment
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
            --hours "$MAX_HOURS" \
            --frontier-model \
            --results-dir "$RESULTS_DIR/$task_name" \
        >> "$LOG_FILE" 2>&1 || true

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "[Worker $WORKER_ID] Finished $task_name in ${ELAPSED}s" | tee -a "$LOG_FILE"

    touch "$result_marker"
    COMPLETED=$((COMPLETED + 1))

    # Check if score improved
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
WORKEREOF
chmod +x "$WORKER_SCRIPT"

# Kill existing session if any
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create tmux session with workers
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Export vars for workers
export NUM_WORKERS

for ((w=0; w<NUM_WORKERS; w++)); do
    GPU_START=$((w * GPUS_PER_WORKER))
    GPU_END=$((GPU_START + GPUS_PER_WORKER - 1))
    if [ "$GPU_END" -gt 7 ]; then GPU_END=7; fi

    ENV_EXPORTS="export NUM_WORKERS=$NUM_WORKERS AMDPILOT_SUPERVISOR_MODEL_URL=$SUPERVISOR_PROXY_URL AMDPILOT_SUPERVISOR_MODEL=claude-opus-4-6"
    WORKER_CMD="$ENV_EXPORTS && bash $WORKER_SCRIPT $w $TASK_LIST $RESULTS_DIR $AMDPILOT_DIR $MODEL_URL $MAX_HOURS $GPU_START $GPU_END"

    if [ "$w" -eq 0 ]; then
        tmux send-keys -t "$SESSION_NAME" "$WORKER_CMD" Enter
    else
        tmux split-window -t "$SESSION_NAME" -h
        tmux send-keys -t "$SESSION_NAME" "$WORKER_CMD" Enter
    fi
done

tmux select-layout -t "$SESSION_NAME" tiled

echo ""
echo "======================================================"
echo "Phase 3 running in tmux session: $SESSION_NAME"
echo "======================================================"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION_NAME        # View live progress"
echo "  tail -f $RESULTS_DIR/worker_*.log   # Monitor logs"
echo "  ls $RESULTS_DIR/*.done | wc -l      # Count completed"
echo ""
