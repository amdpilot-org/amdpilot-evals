#!/bin/bash
RESULTS_DIR="/home/jinpan12/amdpilot/evals/kernelbench/phase3_results"

# Wait for all original workers to finish (check if worker.sh processes are done)
echo "Waiting for original workers to finish..."
while pgrep -f "worker.sh [01] " > /dev/null 2>&1; do
    sleep 10
done
echo "All original workers done!"

# Now launch retries in the tmux panes
echo "Launching retry workers..."

# Worker 0 retry (GPUs 0,1)
tmux send-keys -t kernelbench-phase3:0.0 "bash ${RESULTS_DIR}/launch_offload.sh 0-retry ${RESULTS_DIR}/retry_w0.txt 0 1" Enter

# Worker 1 retry (GPUs 2,3)  
tmux send-keys -t kernelbench-phase3:0.1 "bash ${RESULTS_DIR}/launch_offload.sh 1-retry ${RESULTS_DIR}/retry_w1.txt 2 3" Enter

# Worker 3 retry (GPUs 6,7) — wait for offload worker to finish too
while pgrep -f "offload_w3.txt" > /dev/null 2>&1; do
    sleep 10
done
tmux send-keys -t kernelbench-phase3:0.3 "bash ${RESULTS_DIR}/launch_offload.sh 3-retry ${RESULTS_DIR}/retry_w3.txt 6 7" Enter

echo "All retry workers launched!"
