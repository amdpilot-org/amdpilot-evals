FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315
SHELL ["/bin/bash", "-c"]

# Revert SGLang to the commit just before PR #19935 was merged
RUN cd /sgl-workspace/sglang && \
    git stash && \
    git fetch origin && \
    git checkout 067353f67b56ea6143d9cd77b87b32edde74b6ac && \
    /opt/venv/bin/pip install -e "python[all]" --no-deps 2>/dev/null || true

# Strip git history so the agent cannot look up future commits / the fix
RUN cd /sgl-workspace/sglang && \
    rm -rf .git && \
    git init && \
    git add -A && \
    git commit -m "init" --allow-empty-message 2>/dev/null || true

# Pre-install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy kimi-cli wheels and pre-install so uv caches Python 3.14 + dependencies
COPY dist/kimi-cli/ /tmp/kimi-wheels/
RUN source $HOME/.local/bin/env && \
    uv run --python 3.14 \
      --with /tmp/kimi-wheels/kimi_cli-1.15.0-py3-none-any.whl \
      --with /tmp/kimi-wheels/kosong-0.43.0-py3-none-any.whl \
      --with /tmp/kimi-wheels/pykaos-0.7.0-py3-none-any.whl \
      -- kimi --version

# Copy the test harness to a non-obvious path (orchestrator benchmark command points here)
COPY test_harness.py /opt/test_harness.py
RUN chmod +x /opt/test_harness.py

# Symlink model path
RUN mkdir -p /models && \
    ln -sf /mnt/dcgpuval/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/3367c8d1c68584429fab7faf845a32d5195b6ac1 /models/Kimi-K2.5

WORKDIR /sgl-workspace/sglang
CMD ["sleep", "infinity"]
