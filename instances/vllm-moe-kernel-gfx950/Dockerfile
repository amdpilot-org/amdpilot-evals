FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace

# Clone vllm and checkout to pre-fix state
RUN git clone --depth 100 https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && \
    git fetch --depth 100 origin 7d6917bef552d6aff70142ab9fb8af648081d4db && \
    git checkout -f 7d6917bef552d6aff70142ab9fb8af648081d4db~1

# Install vllm in editable mode (takes ~20 min for C/HIP compilation)
RUN cd /workspace/vllm && \
    pip install --no-build-isolation --no-deps -e . 2>&1 | tail -5

# Clean git history to prevent data leak
RUN cd /workspace/vllm && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

# Symlink python for amdpilot compatibility
RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
