## Stage 1: checkout pre-fix aiter state and clean history
FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326 AS builder

# Roll back aiter to pre-fix state for the blockscale GEMM bug
RUN cd /sgl-workspace/aiter && \
    git fetch --depth 100 origin 3e4552bcc005911ad438de8b09ec5ecc84c03dc7 && \
    git checkout -f 3e4552bcc005911ad438de8b09ec5ecc84c03dc7~1 && \
    rm -f .git/FETCH_HEAD && \
    git checkout --orphan _clean && git add -A && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; git branch -M _clean main && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

# Rebuild aiter C++ extensions from pre-fix source
RUN cd /sgl-workspace/aiter && \
    rm -rf jit/ build/ *.egg-info && \
    /opt/venv/bin/pip install -e . --no-build-isolation --no-deps 2>&1 | tail -5

## Stage 2: fresh base — copy modified aiter, keep sglang untouched
FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326
WORKDIR /workspace

COPY --from=builder /sgl-workspace/aiter /sgl-workspace/aiter

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/aiter /workspace/aiter

CMD ["sleep", "infinity"]
