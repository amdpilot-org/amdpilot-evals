#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMDPILOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
KB_DIR="/home/jinpan12/KernelBench"
DOCKER_IMAGE="rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315"
CONTAINER_NAME="amdpilot-kernelbench"
RUN_NAME="${1:-amdpilot_triton_qwen35_v1}"
MODEL_SERVER="${KERNELBENCH_SERVER_ADDRESS:-10.235.27.218}"
GPU_IDS="${KERNELBENCH_GPU:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

echo "============================================"
echo "KernelBench x amdpilot - Triton on MI355"
echo "============================================"
echo "Run name:     $RUN_NAME"
echo "Model server: $MODEL_SERVER:30000"
echo "GPUs:         $GPU_IDS ($NUM_GPUS devices)"
echo "Docker image: $DOCKER_IMAGE"
echo "============================================"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "[Phase 0] Starting Docker container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size 64g \
    --network host \
    -v "$KB_DIR:/workspace/KernelBench" \
    -v "$AMDPILOT_DIR:/workspace/amdpilot" \
    -e "HIP_VISIBLE_DEVICES=$GPU_IDS" \
    -e "KERNELBENCH_SERVER_ADDRESS=$MODEL_SERVER" \
    -e "SGLANG_API_KEY=dummy" \
    "$DOCKER_IMAGE" \
    sleep infinity

echo "[Phase 0] Installing KernelBench dependencies..."
docker exec "$CONTAINER_NAME" bash -c '
    cd /workspace/KernelBench
    /opt/venv/bin/pip install --no-deps -e . 2>&1 | tail -3
    /opt/venv/bin/pip install pydra-config litellm openai datasets tqdm 2>&1 | tail -5
    echo "Dependencies installed."
'

echo "[Phase 0] Verifying environment..."
docker exec "$CONTAINER_NAME" bash -c '
    /opt/venv/bin/python3 -c "
import torch
print(f\"PyTorch: {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
print(f\"Device count: {torch.cuda.device_count()}\")
if torch.cuda.is_available():
    print(f\"Device 0: {torch.cuda.get_device_name(0)}\")
import triton
print(f\"Triton: {triton.__version__}\")
print(\"Environment OK\")
"
'

for LEVEL in 1 2 3; do
    echo ""
    echo "============================================"
    echo "[Phase 1] Level $LEVEL - Generation (Triton)"
    echo "============================================"
    docker exec "$CONTAINER_NAME" bash -c "
        cd /workspace/KernelBench
        /opt/venv/bin/python3 scripts/generate_samples.py \
            run_name=$RUN_NAME \
            dataset_src=local \
            level=$LEVEL \
            server_type=local \
            model_name=default \
            max_tokens=16384 \
            temperature=0.0 \
            backend=triton \
            num_workers=10 \
            num_samples=1 \
            log_prompt=True \
            verbose=True
    " 2>&1 | tee "$AMDPILOT_DIR/evals/kernelbench/gen_level${LEVEL}.log"

    echo ""
    echo "============================================"
    echo "[Phase 2] Level $LEVEL - Evaluation (Triton on MI355)"
    echo "============================================"
    docker exec "$CONTAINER_NAME" bash -c "
        cd /workspace/KernelBench
        /opt/venv/bin/python3 scripts/eval_from_generations.py \
            run_name=$RUN_NAME \
            dataset_src=local \
            level=$LEVEL \
            backend=triton \
            gpu_arch='[\"gfx950\"]' \
            num_gpu_devices=$NUM_GPUS \
            timeout=300 \
            num_correct_trials=5 \
            num_perf_trials=100 \
            measure_performance=True
    " 2>&1 | tee "$AMDPILOT_DIR/evals/kernelbench/eval_level${LEVEL}.log"

    echo "[Level $LEVEL] Generation and evaluation complete."
done

echo ""
echo "============================================"
echo "[Phase 3] Analysis"
echo "============================================"
docker exec "$CONTAINER_NAME" bash -c "
    cd /workspace/KernelBench
    /opt/venv/bin/python3 /workspace/amdpilot/evals/kernelbench/analyze_results.py \
        --run-name $RUN_NAME \
        --runs-dir /workspace/KernelBench/runs \
        --output /workspace/amdpilot/evals/kernelbench/results_${RUN_NAME}.json
"

echo ""
echo "============================================"
echo "All levels complete! Results in:"
echo "  Kernels: $KB_DIR/runs/$RUN_NAME/"
echo "  Analysis: $AMDPILOT_DIR/evals/kernelbench/results_${RUN_NAME}.json"
echo "============================================"
