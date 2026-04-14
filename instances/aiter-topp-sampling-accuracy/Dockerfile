FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326

# The base image (Mar 26, 2026) post-dates the fix merge (Feb 12, 2026),
# so aiter in the image already contains the fix. We must revert aiter
# to the commit immediately before the fix to reproduce the bug.

ARG MERGE_COMMIT=d831d658d05d1e718d53e696a561b91122426594

RUN cd /sgl-workspace/aiter \
    && git fetch --depth 100 origin ${MERGE_COMMIT} \
    && git checkout -f ${MERGE_COMMIT}~1 \
    && pip install -e . --no-build-isolation --no-deps

# Clean up git history to avoid leaking the fix commit to the agent
RUN cd /sgl-workspace/aiter \
    && rm -rf .git

# Copy eval harness
COPY test_harness.py /eval/test_harness.py
COPY task_description.md /eval/task_description.md

WORKDIR /sgl-workspace/aiter
