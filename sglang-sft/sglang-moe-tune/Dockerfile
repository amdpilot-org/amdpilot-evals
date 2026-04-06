FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315
SHELL ["/bin/bash", "-c"]

# Checkout SGLang at the pre-fix commit for PR #19228
RUN cd /sgl-workspace/sglang && \
    git stash && \
    git fetch origin && \
    git checkout 288300aafd15c5a454404d44b312de4a019f59b3 && \
    /opt/venv/bin/pip install -e "python[all]" --no-deps 2>/dev/null || true

# Strip git history to prevent answer peeking
RUN cd /sgl-workspace/sglang && \
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

# Create /workspace and copy task files
RUN mkdir -p /workspace && ln -sf /sgl-workspace/sglang /workspace/sglang
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN chmod +x /workspace/test_harness.py

# Symlink models
RUN mkdir -p /sgl-workspace/models && \
    ln -sf /mnt/dcgpuval/huggingface/hub /sgl-workspace/models/hub 2>/dev/null || true

WORKDIR /workspace
CMD ["sleep", "infinity"]
