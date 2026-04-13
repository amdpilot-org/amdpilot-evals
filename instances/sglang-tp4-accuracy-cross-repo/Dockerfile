FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326

WORKDIR /workspace

# PR #18607: TP=4 accuracy degradation on AMD GPUs.
# Merge commit: e20e6c28b9bf56f632c64fdfeb5b8871237e8d88
# Checkout to merge_commit~1 to get the pre-fix (buggy) state.
RUN cd /sgl-workspace/sglang && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin e20e6c28b9bf56f632c64fdfeb5b8871237e8d88 && \
    git checkout -f e20e6c28b9bf56f632c64fdfeb5b8871237e8d88~1 && \
    /opt/venv/bin/pip install --no-deps -e python/

# Clean git history to prevent data leak
RUN cd /sgl-workspace/sglang && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

# Ensure workspace symlink exists
RUN ln -sfn /sgl-workspace/sglang /workspace/sglang

# Copy eval artifacts
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
