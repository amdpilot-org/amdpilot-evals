#!/bin/bash
set -euo pipefail

cd /workspace/openpi

if [ -f /workspace/bench_config.env ]; then
    # shellcheck disable=SC1091
    source /workspace/bench_config.env
fi

if [ ! -f .venv/bin/python ]; then
    echo "[ERROR] .venv/bin/python not found. Run: cd /workspace/openpi && uv sync"
    exit 1
fi

CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-/root/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch}"
JAX_CHECKPOINT_DIR="/root/.cache/openpi/openpi-assets/checkpoints/pi0_libero"
NORM_STATS_PATH="${CHECKPOINT_DIR}/assets/physical-intelligence/libero/norm_stats.json"

if [ ! -f "${CHECKPOINT_DIR}/model.safetensors" ] || [ ! -f "${NORM_STATS_PATH}" ]; then
    echo "[INFO] PyTorch checkpoint assets incomplete at ${CHECKPOINT_DIR}; preparing from OpenPI assets"
    .venv/bin/python - <<'PY'
from openpi.shared import download

path = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
print(f"[INFO] Downloaded OpenPI JAX checkpoint to {path}")
PY
    if [ ! -f "${CHECKPOINT_DIR}/model.safetensors" ]; then
        .venv/bin/python examples/convert_jax_model_to_pytorch.py \
            --checkpoint_dir "${JAX_CHECKPOINT_DIR}" \
            --config_name pi0_libero \
            --output_path "${CHECKPOINT_DIR}"
    fi
    rm -rf "${CHECKPOINT_DIR}/assets"
    cp -r "${JAX_CHECKPOINT_DIR}/assets" "${CHECKPOINT_DIR}/assets"
fi

OUTPUT=$(.venv/bin/python benchmark_pi0_libero_rocm.py \
    --config pi0_libero \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device cuda:0 \
    --num-warmup 3 \
    --num-runs 20 \
    2>&1) || true

echo "$OUTPUT"
echo "$OUTPUT" > /workspace/bench_output.log

METRIC=""
METRIC=$(echo "$OUTPUT" | grep -oP 'Throughput\s*:\s*\K[\d.]+(?=\s*inf/s)' | tail -1) || true

if [ -z "$METRIC" ]; then
    METRIC=$(echo "$OUTPUT" | grep -oP 'throughput[_\s]*inf[_/]s\w*\s*[:=]\s*\K[\d.]+' | tail -1) || true
fi

if [ -z "$METRIC" ]; then
    METRIC=$(echo "$OUTPUT" | grep -oP '[\d.]+(?=\s*inferences?/sec)' | tail -1) || true
fi

if [ -z "$METRIC" ]; then
    METRIC=$(echo "$OUTPUT" | grep -oP 'fps=\K[\d.]+' | tail -1) || true
fi

if [ -n "$METRIC" ]; then
    echo ""
    echo "THROUGHPUT: ${METRIC} inf/s"
    echo "throughput_inf_per_sec: ${METRIC}"
else
    echo ""
    echo "[WARN] Could not extract throughput metric from benchmark output"
    echo "THROUGHPUT: 0.0 inf/s"
    echo "throughput_inf_per_sec: 0.0"
fi
