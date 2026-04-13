## Stage 1: clone repo at pre-fix state and clean git history
FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330 AS builder

RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3
RUN git clone --depth 100 https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && \
    git fetch --depth 100 origin 1b117cb0ac51b84dd7aed364e6e802dea5147ca6 && \
    git checkout -f 1b117cb0ac51b84dd7aed364e6e802dea5147ca6~1 && \
    rm -f .git/FETCH_HEAD && \
    git checkout --orphan _clean && git add -A && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; git branch -M _clean main && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

## Stage 2: fresh base — copy cleaned repo and install
FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace

RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3

COPY --from=builder /workspace/vllm /workspace/vllm

RUN cd /workspace/vllm && \
    pip install --no-build-isolation --no-deps -e .

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
