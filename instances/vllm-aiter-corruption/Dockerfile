ARG BASE_IMAGE=rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Pin to the parent of the fix commit for PR #37228 (vllm#35925).
# Fix commit: a4cf9b22ba8f08df8d7977c2b0753fe20231cffc
# Parent:     9c3ae04bfe6532d6d6d812ae6688366c9911555b
# This ensures the AITER corruption bug is present for the agent to fix.
#
# The base image may or may not have /workspace/vllm/.git — handle both cases.
RUN if [ ! -d /workspace/vllm/.git ]; then \
        mkdir -p /workspace/vllm && cd /workspace/vllm && git init; \
    fi && \
    cd /workspace/vllm && \
    (git remote set-url origin https://github.com/vllm-project/vllm.git 2>/dev/null || \
     git remote add origin https://github.com/vllm-project/vllm.git) && \
    git fetch --depth 100 origin a4cf9b22ba8f08df8d7977c2b0753fe20231cffc && \
    git checkout -f a4cf9b22ba8f08df8d7977c2b0753fe20231cffc~1 && \
    pip install --no-build-isolation --no-deps -e . && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    cd /workspace/vllm && \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

ENV PYTHONPATH=/workspace/vllm
RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN printf '#!/bin/bash\npgrep -f "python3 -m (sglang|vllm)" | xargs -r kill -9 2>/dev/null\nsleep 3\n' \
    > /usr/local/bin/safe-kill-server && chmod +x /usr/local/bin/safe-kill-server

CMD ["sleep", "infinity"]
