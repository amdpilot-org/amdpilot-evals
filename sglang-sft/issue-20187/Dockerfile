FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315
SHELL ["/bin/bash", "-c"]

# Revert SGLang to the commit just before PR #20187 was merged
RUN cd /sgl-workspace/sglang && \
    git stash && \
    git fetch origin && \
    git checkout 751c4540990152bc045d44859ca1498d34337750 && \
    /opt/venv/bin/pip install -e "python[all]" --no-deps 2>/dev/null || true

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

# Copy the test harness
COPY test_harness.py /workspace/test_harness.py
RUN chmod +x /workspace/test_harness.py

# Symlink model path
RUN mkdir -p /models && \
    ln -sf /mnt/dcgpuval/datasets/data/DeepSeek-R1-0528-MXFP4-Preview /models/DeepSeek-R1-MXFP4-Preview

WORKDIR /sgl-workspace/sglang
CMD ["sleep", "infinity"]
