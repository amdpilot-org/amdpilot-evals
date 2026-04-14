## Stage 1: check out the target commit in a throwaway builder
FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330 AS builder
WORKDIR /workspace
RUN git clone --depth 100 https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && \
    git fetch --depth 100 origin 679c6a3eccfadb5f80957801f1b6a0c9e8cc9b41 && \
    git checkout -f 679c6a3eccfadb5f80957801f1b6a0c9e8cc9b41~1 && \
    git checkout --orphan _clean && git add -A && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; git branch -M _clean main && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

## Stage 2: fresh base — copy only the cleaned repo
FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace
RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3
COPY --from=builder /workspace/vllm /workspace/vllm
RUN cd /workspace/vllm && pip install --no-build-isolation --no-deps -e .
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
