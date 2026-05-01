FROM rocm/sgl-dev:v0.5.10rc0-rocm720-mi30x-20260420

WORKDIR /workspace

ENV PATH="/root/.local/bin:${PATH}" \
    GIT_LFS_SKIP_SMUDGE=1 \
    TOKENIZERS_PARALLELISM=false \
    GPU_COREDUMP_ENABLE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      ca-certificates \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN rm -rf /workspace/openpi \
    && git clone https://github.com/amdpilot-org/openpi.git /workspace/openpi \
    && cd /workspace/openpi \
    && git checkout e4429ad35ec380842dc72b4074735cf3e8a503c2

# openpi#3's benchmark script is not present in amdpilot-org/openpi@e4429ad.
# Fetch the canonical script from the issue-provided benchmark source before
# baseline verification, otherwise the baseline emits no metric.
RUN curl -LsSf -o /workspace/openpi/benchmark_pi0_libero_rocm.py \
      https://raw.githubusercontent.com/prasad-nair-amd/openpi/main/benchmark_pi0_libero_rocm.py

RUN cd /workspace/openpi \
    && uv sync 2>&1 || echo "[WARN] uv sync had issues, benchmark will retry at runtime"

# OpenPI may resolve non-ROCm wheels by default. Force ROCm PyTorch into the uv venv.
RUN cd /workspace/openpi \
    && uv pip install --python .venv/bin/python --no-cache-dir --force-reinstall \
      --index-url https://download.pytorch.org/whl/rocm7.2 torch torchvision

# Keep the transformers replacement overlay used by the prior baseline-passing run.
RUN cd /workspace/openpi \
    && TRANSFORMERS_DIR="$(.venv/bin/python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')" \
    && if [ -d ./src/openpi/models_pytorch/transformers_replace ]; then \
         cp -r ./src/openpi/models_pytorch/transformers_replace/* "${TRANSFORMERS_DIR}/"; \
       fi

COPY bench_openpi.sh /workspace/bench_openpi.sh
COPY bench_pi0_libero.sh /workspace/bench_pi0_libero.sh
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN chmod +x /workspace/bench_openpi.sh /workspace/bench_pi0_libero.sh /workspace/test_harness.py

ENTRYPOINT []
CMD ["sleep", "infinity"]
