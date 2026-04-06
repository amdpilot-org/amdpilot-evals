ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Issue sglang#20691: bug introduced by PR #19122 (Feb 27), likely fixed by
# PR #20409 (Mar 18) or #21315 (Mar 30) which changed aiter_backend.py.
# Pin to Mar 16 commit — after bug introduction, before fixes.
# Main HEAD has the fix, so we CANNOT use latest main.
RUN cd /sgl-workspace/sglang && \
    git fetch origin && \
    git checkout -- . && \
    git checkout 1eea744855eff823637053d74bf51f5d1a758e07

# Ensure workspace symlink exists
RUN ln -sfn /sgl-workspace/sglang /workspace/sglang

# Copy eval artifacts
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

# Structural safety: safe kill script (lesson #4 from vllm-run-lessons.md)
RUN printf '#!/bin/bash\npgrep -f "python3 -m sglang" | xargs -r kill -9\nsleep 3\n' \
    > /usr/local/bin/safe-kill-server && chmod +x /usr/local/bin/safe-kill-server

CMD ["sleep", "infinity"]
