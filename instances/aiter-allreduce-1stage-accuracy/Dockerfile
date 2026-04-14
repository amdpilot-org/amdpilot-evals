FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326
WORKDIR /workspace

# The base image (Mar 26, 2026) may post-date the fix merge, so aiter in
# the image may already contain the fix. Checkout to the commit immediately
# before the fix to reproduce the bug.
ARG MERGE_COMMIT=43b7379b895ffc7545952d5835dd573b20f22d67

RUN cd /sgl-workspace/aiter && \
    git fetch --depth 100 origin ${MERGE_COMMIT} && \
    git checkout -f ${MERGE_COMMIT}~1 && \
    pip install -e . --no-build-isolation --no-deps

# Clean up git to prevent the executor from finding the fix via history.
RUN cd /sgl-workspace/aiter && \
    rm -rf .git

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/aiter /workspace/aiter

CMD ["sleep", "infinity"]
