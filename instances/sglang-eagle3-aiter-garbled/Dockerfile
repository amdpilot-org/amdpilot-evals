FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226

WORKDIR /workspace

# PR #19362: EAGLE3 spec decode garbled on non-MLA models.
# Merge commit: 441045a7bf2496cef0130b03ac88c74eb1cb8e8a
# Checkout to merge_commit~1 to get the pre-fix (buggy) state.
RUN rm -rf /sgl-workspace/sglang && \
    git clone --depth 1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git fetch --depth 100 origin 441045a7bf2496cef0130b03ac88c74eb1cb8e8a && \
    git checkout -f 441045a7bf2496cef0130b03ac88c74eb1cb8e8a~1 && \
    /opt/venv/bin/pip install --no-deps -e python/ && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

# Ensure workspace symlinks exist
RUN ln -sfn /sgl-workspace/sglang /workspace/sglang && \
    ln -sfn /sgl-workspace/aiter /workspace/aiter 2>/dev/null; true

# Copy eval artifacts
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
