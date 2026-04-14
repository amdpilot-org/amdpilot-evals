## Stage 1: generate fixtures and prepare buggy repo in a throwaway builder
FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317 AS builder

# Generate reference fixtures on CPU (no GPU needed)
COPY gen_fixtures.py /tmp/gen_fixtures.py
RUN /opt/venv/bin/python3 /tmp/gen_fixtures.py /tmp/mxfp4_fixtures/ && \
    rm /tmp/gen_fixtures.py

# Checkout to pre-fix state and clean git history
RUN cd /sgl-workspace/aiter && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin 584e490b748479816136158075c2b20fd5f6c03a && \
    git checkout -f 584e490b748479816136158075c2b20fd5f6c03a~1 && \
    rm -f .git/FETCH_HEAD && \
    git checkout --orphan _clean && git add -A && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; git branch -M _clean main && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

## Stage 2: fresh base — copy only the modified repo and fixtures
FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace

COPY --from=builder /sgl-workspace/aiter /sgl-workspace/aiter
COPY --from=builder /tmp/mxfp4_fixtures/ /workspace/fixtures/

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/aiter /workspace/aiter

CMD ["sleep", "infinity"]
