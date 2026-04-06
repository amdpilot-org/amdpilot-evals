ARG BASE_IMAGE=rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Historical bundle reconstructed from the successful 35925 rerun materials.
# Commit-window tightening is still a follow-up item.
RUN if [ -d /workspace/vllm/.git ]; then \
        cd /workspace/vllm && \
        (git remote set-url origin https://github.com/vllm-project/vllm.git 2>/dev/null || \
         git remote add origin https://github.com/vllm-project/vllm.git) && \
        git fetch origin main && \
        git reset --hard origin/main; \
    else \
        git clone --depth 1 --branch main https://github.com/vllm-project/vllm.git /workspace/vllm; \
    fi

RUN pip uninstall -y vllm 2>/dev/null || true
ENV PYTHONPATH=/workspace/vllm
RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN printf '#!/bin/bash\npgrep -f "python3 -m (sglang|vllm)" | xargs -r kill -9 2>/dev/null\nsleep 3\n' \
    > /usr/local/bin/safe-kill-server && chmod +x /usr/local/bin/safe-kill-server

CMD ["sleep", "infinity"]
