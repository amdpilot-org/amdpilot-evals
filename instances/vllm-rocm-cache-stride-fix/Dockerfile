FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace
RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3
RUN cd /workspace/vllm && \
    git remote set-url origin https://github.com/vllm-project/vllm.git 2>/dev/null || \
    git remote add origin https://github.com/vllm-project/vllm.git && \
    git fetch --depth 100 origin a4cf9b22ba8f08df8d7977c2b0753fe20231cffc && \
    git checkout -f a4cf9b22ba8f08df8d7977c2b0753fe20231cffc~1 && \
    pip install --no-build-isolation --no-deps -e . && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
