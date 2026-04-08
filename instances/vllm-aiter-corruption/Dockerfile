ARG BASE_IMAGE=rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
FROM ${BASE_IMAGE}

WORKDIR /workspace
RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3

# Pin to the parent of the fix commit for PR #37228 (vllm#35925).
# Fix commit: a4cf9b22ba8f08df8d7977c2b0753fe20231cffc
# Parent:     9c3ae04bfe6532d6d6d812ae6688366c9911555b
# This ensures the AITER corruption bug is present for the agent to fix.
#
# The base image may or may not have /workspace/vllm/.git — handle both cases.
# Also prebuild the vLLM native extensions into the source tree so runtime trials
# cannot go green by spending their budget on bootstrap work.
RUN if [ ! -d /workspace/vllm/.git ]; then \
        mkdir -p /workspace/vllm && cd /workspace/vllm && git init; \
    fi && \
    cd /workspace/vllm && \
    (git remote set-url origin https://github.com/vllm-project/vllm.git 2>/dev/null || \
     git remote add origin https://github.com/vllm-project/vllm.git) && \
    git fetch --depth 100 origin a4cf9b22ba8f08df8d7977c2b0753fe20231cffc && \
    git checkout -f a4cf9b22ba8f08df8d7977c2b0753fe20231cffc~1 && \
    (/usr/bin/python3 -m pip uninstall -y vllm 2>/dev/null || true) && \
    VLLM_TARGET_DEVICE=rocm MAX_JOBS=4 \
        /usr/bin/python3 -m pip install --no-build-isolation --no-deps -e . && \
    VLLM_TARGET_DEVICE=rocm MAX_JOBS=4 \
        /usr/bin/python3 setup.py build_ext --inplace && \
    find build -type f -path '*/vllm/*.so' -exec cp -f {} vllm/ \; && \
    /usr/bin/python3 -c "from pathlib import Path; root=Path('/workspace/vllm/vllm'); needed=['_C.abi3.so', '_moe_C.abi3.so', '_rocm_C.abi3.so', 'cumem_allocator.abi3.so']; missing=[name for name in needed if not (root / name).exists()]; assert not missing, f'Missing vLLM extensions: {missing}'" && \
    rm -rf build && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    cd /workspace/vllm && \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

ENV PYTHONPATH=/workspace/vllm

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN printf '#!/bin/bash\npgrep -f "python3 -m (sglang|vllm)" | xargs -r kill -9 2>/dev/null\nsleep 3\n' \
    > /usr/local/bin/safe-kill-server && chmod +x /usr/local/bin/safe-kill-server

CMD ["sleep", "infinity"]
