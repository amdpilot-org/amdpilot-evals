# openpi pi0 LIBERO inference latency optimization on AMD ROCm.
# Source issue: https://github.com/prasad-nair-amd/openpi/issues/1
#
# Base is the standard sgl-dev image used by every other amdpilot eval so
# agents operate in a familiar environment with aiter, triton, etc. available.
# openpi itself runs out of an isolated .venv (Python 3.11 managed by uv) to
# satisfy openpi's pyproject (requires-python >= 3.11) without disturbing
# the sgl-dev /opt/venv Python 3.10.
#
# Build:
#   podman build -t amdpilot-eval-openpi-pi0-libero-optimize .

FROM docker.io/rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260310

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/root/.local/bin:$PATH \
    GIT_LFS_SKIP_SMUDGE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-pythons \
    ROCM_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 \
    TORCH_VERSION=2.11.0

# git-lfs is required by openpi (model asset handling even with SMUDGE skip);
# build-essential is already in sgl-dev. Install uv and keep it pinned.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN curl -Lsf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv python install 3.11

WORKDIR /workspace

# Clone the full openpi runtime package (Physical-Intelligence) and overlay the
# benchmark scripts from prasad-nair-amd/openpi on top.
RUN git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git /workspace/openpi \
    && git clone --depth 1 https://github.com/prasad-nair-amd/openpi.git /tmp/openpi-bench \
    && cp /tmp/openpi-bench/benchmark_pi0_libero_rocm.py /workspace/openpi/ \
    && cp /tmp/openpi-bench/benchmark_pi0_libero_cuda.py /workspace/openpi/ \
    && cp /tmp/openpi-bench/setup_mi300x.sh /workspace/openpi/ \
    && cp /tmp/openpi-bench/BENCHMARK_SETUP.md /workspace/openpi/ \
    && rm -rf /tmp/openpi-bench

WORKDIR /workspace/openpi

# openpi pyproject requires Python >=3.11; sgl-dev only ships 3.10, so we
# pin uv to its managed 3.11. uv sync does not seed pip — we ensurepip
# afterwards so the ROCm torch force-reinstall can run.
RUN cd /workspace/openpi \
    && GIT_LFS_SKIP_SMUDGE=1 /root/.local/bin/uv sync --python 3.11 2>&1 | tail -5 \
    && /workspace/openpi/.venv/bin/python -m ensurepip --upgrade \
    && /workspace/openpi/.venv/bin/python -m pip install --no-cache-dir --upgrade pip

# Force-install the ROCm torch wheel over the CUDA torch that uv pulled.
# Wheel bundles its own libamdhip64 / librccl so it works on sgl-dev's ROCm 7.0
# runtime as long as the host kernel/KFD supports gfx942 (MI300X does).
RUN /workspace/openpi/.venv/bin/pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" torchvision \
        --index-url "${ROCM_PYTORCH_INDEX_URL}" --force-reinstall 2>&1 | tail -10

RUN /workspace/openpi/.venv/bin/python -c \
    "import torch; hip=getattr(torch.version,'hip',None); \
     assert hip is not None, 'torch is not ROCm build: '+torch.__version__; \
     print('torch:', torch.__version__, 'HIP:', hip)"

# Commit the repo state so agent-made changes show up as a git diff
# (extract_patch in the RecursiveNode framework relies on this).
RUN cd /workspace/openpi \
    && git config user.email "build@eval" \
    && git config user.name "eval" \
    && git add -A \
    && git commit -m "baseline: openpi + ROCm torch + benchmark scripts" --allow-empty 2>&1 | tail -3

# Test harness and task description live at /workspace (not inside repo).
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
COPY run_benchmark.sh /workspace/run_benchmark.sh
RUN chmod +x /workspace/test_harness.py /workspace/run_benchmark.sh

ENV OPENPI_DIR=/workspace/openpi \
    OPENPI_CHECKPOINT_DIR=/root/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch \
    OPENPI_CONFIG_NAME=pi0_libero \
    OPENPI_NUM_WARMUP=3 \
    OPENPI_NUM_RUNS=30

CMD ["sleep", "infinity"]
