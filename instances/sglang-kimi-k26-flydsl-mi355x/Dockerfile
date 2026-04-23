# Kimi-K2.6 MI355X FlyDSL optimization eval.
#
# Base image is the public Docker Hub tag from PR #23381:
#   https://github.com/sgl-project/sglang/pull/23381
# which ships SGLang v0.5.10rc0 + ROCm 7.2.0 + the PR's tuned Triton
# fused-MoE JSON config (E=384,N=256,MI355X,int4_w4a16) + the
# SGLANG_DEEPSEEK_LOAD_MAX_WORKERS env var. Kimi-K2.6 weights are baked
# in at /sgl-workspace/models/Kimi-K2.6.
#
# On top of that baseline we layer:
#  1. FlyDSL main (Python DSL + MLIR-built compiler) at /opt/FlyDSL
#  2. AITER dev/kimi-K2.5 branch (carries flydsl_moe_stage1/stage2 hooks)
#  3. amdpilot-org fork remotes for patch delivery
#  4. The bench wrapper, accuracy harness, and task description
#
# The container starts with FlyDSL DISABLED via task.yaml env so Phase 1
# reproduces PR #23381's baseline decode tok/s numbers. Trials then flip
# AITER_USE_FLYDSL_MOE / _STAGE1 / _STAGE2 / FLYDSL_W4A16_HYBRID inside
# /workspace/bench_config.env before each verification run.

ARG BASE_IMAGE=jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420
FROM ${BASE_IMAGE}

WORKDIR /workspace

# --- MLIR (FlyDSL compile-time dependency) ------------------------------------
# FlyDSL needs MLIR with Python bindings for all dialects (ROCDL, vector,
# llvm, ExecutionEngine). The public base image
# `jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420` ships ROCm 7.2 with
# `rocm-llvm-dev` installable (provides MLIRConfig.cmake), but the ROCm LLVM
# build lacks the MLIRPythonSources.* targets FlyDSL requires. Therefore we
# must build LLVM+MLIR from upstream source via FlyDSL's own
# `scripts/build_llvm.sh`. This is a one-time ~30-45 min Docker-build cost;
# result lives under /opt/FlyDSL/../llvm-project/mlir_install and is reused
# across every trial once the image is built.
ENV MLIR_PATH=/opt/llvm-project/mlir_install

# --- Build deps + Python bindings package -----------------------------------
# `rocm-llvm-dev` is installed so rocm-gpu target headers are available to
# FlyDSL's final build (it needs them for gfx950 codegen). `nanobind`
# (>=2.9) is required by MLIR's Python binding codegen and by FlyDSL's
# own C-extension glue.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        cmake ninja-build rocm-llvm-dev; \
    rm -rf /var/lib/apt/lists/*; \
    /opt/venv/bin/pip install --no-cache-dir 'nanobind>=2.9' numpy pybind11

# --- FlyDSL: clone + build LLVM/MLIR + build FlyDSL -------------------------
# `scripts/build_llvm.sh` shallow-clones upstream llvm-project at the pinned
# commit from `thirdparty/llvm-hash.txt`, configures with
# `-DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMLIR_ENABLE_ROCM_RUNNER=ON`, builds
# ~1500 targets (30-45 min), and installs under /opt/llvm-project/mlir_install.
# Then `scripts/build.sh` builds FlyDSL's own fly-dialect compiler (~5 min)
# against that MLIR. Finally `pip install -e python/` wires the Python
# package into the venv.
#
# Note: /opt/venv/bin/python3 is prepended to PATH so build_llvm.sh's
# internal `pip install nanobind numpy pybind11` lands in the right venv.
# build_llvm.sh derives BASE_DIR from the parent of FlyDSL's repo root and
# expects to create /opt/llvm-project itself via `git init`; do NOT
# pre-mkdir that path or the script will error with "not a git repository".
#
# Patch `-DMLIR_ENABLE_ROCM_RUNNER=ON` -> `OFF`. MLIR's ROCm runner is a
# JIT-execution component (runs MLIR ops directly on GPU), NOT the
# AMDGPU-codegen path FlyDSL needs. Its cmake probe requires `amdgpu-arch`
# to talk to the GPU at configure time, which Docker `build` (no GPU) can't
# provide. FlyDSL still gets AMDGPU codegen via `LLVM_TARGETS_TO_BUILD=AMDGPU`,
# which the same cmake line already enables.
RUN set -eux; \
    git clone --depth 1 https://github.com/ROCm/FlyDSL.git /opt/FlyDSL; \
    cd /opt/FlyDSL; \
    sed -i 's/-DMLIR_ENABLE_ROCM_RUNNER=ON/-DMLIR_ENABLE_ROCM_RUNNER=OFF/' scripts/build_llvm.sh; \
    grep MLIR_ENABLE_ROCM_RUNNER scripts/build_llvm.sh; \
    export PATH=/opt/venv/bin:$PATH; \
    export LLVM_INSTALL_DIR=/opt/llvm-project/mlir_install; \
    bash scripts/build_llvm.sh -j$(($(nproc) / 2)); \
    export MLIR_PATH=/opt/llvm-project/mlir_install; \
    bash scripts/build.sh -j$(($(nproc) / 2)); \
    /opt/venv/bin/pip install -e .
ENV DSL2_ROOT=/opt/FlyDSL
ENV PYTHONPATH=/opt/FlyDSL/build-fly/python_packages:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/FlyDSL/build-fly/lib:/opt/llvm-project/mlir_install/lib:${LD_LIBRARY_PATH}

# --- AITER dev/kimi-K2.5 branch ----------------------------------------------
# Carries the flydsl_moe_stage1 / flydsl_moe_stage2 hooks that the
# blog's env vars (AITER_USE_FLYDSL_MOE_STAGE1, AITER_USE_FLYDSL_MOE_STAGE2,
# FLYDSL_W4A16_HYBRID, AITER_ENFORCE_DSL, etc.) gate on. K2.6 reuses the
# K2.5 MoE/MLA architecture per PR #23381 ("K2.6 reuses the K2.5 MoE/MLA
# architecture, 384 experts, W4A16"), so these kernels should apply
# without modification — Phase A in task_description.md confirms this
# on the actual K2.6 weights.
RUN set -eux; \
    cd /sgl-workspace/aiter; \
    git fetch origin dev/kimi-K2.5; \
    git checkout dev/kimi-K2.5; \
    /opt/venv/bin/pip install -e . 2>&1 | tail -5

# --- Fork remotes for patch delivery -----------------------------------------
# DeliveryConfig default AMDPILOT_FORK=amdpilot-org/sglang routes pushes
# to the amdpilot-org fork. The final patch lands on branch
# `amdpilot/sglang-kimi-k26-flydsl-mi355x` once trials converge.
RUN set -eux; \
    cd /sgl-workspace/sglang; \
    git remote add fork git@github.com:amdpilot-org/sglang.git 2>/dev/null || true; \
    cd /sgl-workspace/aiter; \
    git remote add fork git@github.com:amdpilot-org/aiter.git 2>/dev/null || true
ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

# --- Eval artifacts ----------------------------------------------------------
COPY bench_flydsl_k26.sh /workspace/bench_flydsl_k26.sh
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN chmod +x /workspace/bench_flydsl_k26.sh

# Pre-seed an empty bench_config.env so bench_flydsl_k26.sh can `source`
# it unconditionally. Agents append FlyDSL knobs / extra server flags here
# during trials.
RUN touch /workspace/bench_config.env

CMD ["sleep", "infinity"]
