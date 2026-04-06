FROM rocm/vllm-dev:nightly
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace

# Clone vllm at the pre-fix commit
RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && git checkout 9251ed5c4fc6c954a5cdc5399d9d4f25ea5a8dd3

# Build vllm from source (required for the CUDA kernel changes)
RUN cd /workspace/vllm && \
    VLLM_TARGET_DEVICE=rocm MAX_JOBS=8 \
    pip install -e . --no-build-isolation 2>&1 | tail -50

# Remove nightly .so files to ensure agent-compiled version is used
RUN rm -f /usr/local/lib/python3.12/dist-packages/vllm/_rocm_C*.so \
         /usr/local/lib/python3.12/dist-packages/vllm/_C*.so \
         /usr/local/lib/python3.12/dist-packages/vllm/_moe_C*.so \
         /usr/local/lib/python3.12/dist-packages/vllm/cumem_allocator*.so

# Strip git history to prevent answer peeking
RUN cd /workspace/vllm && \
    rm -rf .git && \
    git init && \
    git add -A && \
    git commit -m "init" --allow-empty-message 2>/dev/null || true

# Pre-install uv + kimi-cli
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
COPY dist/kimi-cli/ /tmp/kimi-wheels/
RUN source $HOME/.local/bin/env && \
    uv run --python 3.14 \
      --with /tmp/kimi-wheels/kimi_cli-1.15.0-py3-none-any.whl \
      --with /tmp/kimi-wheels/kosong-0.43.0-py3-none-any.whl \
      --with /tmp/kimi-wheels/pykaos-0.7.0-py3-none-any.whl \
      -- kimi --version

# Copy task files
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN chmod +x /workspace/test_harness.py

CMD ["sleep", "infinity"]
