FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace

# Checkout pre-fix state
ARG MERGE_COMMIT=a4cf9b22ba8f08df8d7977c2b0753fe20231cffc
RUN cd /workspace/vllm && \
    git fetch --depth 100 origin ${MERGE_COMMIT} && \
    git checkout -f ${MERGE_COMMIT}~1 && \
    pip install -e . --no-build-isolation --no-deps 2>/dev/null || true

# Clean git history
RUN cd /workspace/vllm && rm -rf .git

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
