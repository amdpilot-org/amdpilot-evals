FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326
WORKDIR /workspace

# Checkout pre-fix state
ARG MERGE_COMMIT=17721b00fddfb85e21af80e0b23ef249d1bfe46a
RUN cd /sgl-workspace/sglang && \
    git fetch --depth 100 origin ${MERGE_COMMIT} && \
    git checkout -f ${MERGE_COMMIT}~1 && \
    pip install -e "python[all]" --no-build-isolation --no-deps 2>/dev/null || true

# Clean git history to prevent executor from finding the fix
RUN cd /sgl-workspace/sglang && rm -rf .git

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/sglang /workspace/sglang

CMD ["sleep", "infinity"]
