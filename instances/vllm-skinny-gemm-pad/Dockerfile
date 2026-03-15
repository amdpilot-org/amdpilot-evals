ARG BASE_IMAGE=rocm/vllm-dev:nightly
FROM ${BASE_IMAGE}
WORKDIR /workspace

# Clone vllm source at pre-fix commit using treeless clone
RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && git checkout 9251ed5c4fc6c954a5cdc5399d9d4f25ea5a8dd3

# Build vllm from source so the compiled kernel matches the pre-fix source.
# VLLM_TARGET_DEVICE=rocm is required for older commits that try to fetch
# precompiled wheels from AMD PyPI (which fails for this commit).
# MAX_JOBS limits parallel compilation to avoid OOM during docker build.
RUN cd /workspace/vllm && \
    VLLM_TARGET_DEVICE=rocm MAX_JOBS=8 \
    pip install -e . --no-build-isolation 2>&1 | tail -50

# Remove any leftover nightly .so to guarantee the agent-compiled version is used
RUN rm -f /usr/local/lib/python3.12/dist-packages/vllm/_rocm_C*.so \
         /usr/local/lib/python3.12/dist-packages/vllm/_C*.so \
         /usr/local/lib/python3.12/dist-packages/vllm/_moe_C*.so \
         /usr/local/lib/python3.12/dist-packages/vllm/cumem_allocator*.so

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
