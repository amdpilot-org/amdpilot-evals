ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Pin SGLang to a commit where issue #21614 (gibberish at high concurrency) is present.
# Commit from ~2026-03-25, before the issue was closed on 2026-03-31.
RUN cd /sgl-workspace/sglang && \
    git fetch origin && \
    git checkout -- . && \
    git checkout 3937a07b177274da508cb33391fff50075f69347

# Ensure workspace symlink exists
RUN ln -sfn /sgl-workspace/sglang /workspace/sglang

# Install httpx for async concurrency testing in the harness
RUN /opt/venv/bin/pip install httpx 2>/dev/null || true

# Copy eval artifacts
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

# Structural safety: safe kill script (lesson #4 from vllm-run-lessons.md)
RUN printf '#!/bin/bash\npgrep -f "python3 -m sglang" | xargs -r kill -9\nsleep 3\n' \
    > /usr/local/bin/safe-kill-server && chmod +x /usr/local/bin/safe-kill-server

CMD ["sleep", "infinity"]
