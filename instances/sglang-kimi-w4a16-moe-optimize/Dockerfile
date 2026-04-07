ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}
WORKDIR /workspace

# Check out sglang at pre-fix state (parent of PR #17863 merge commit).
# At this commit, W4A16 MoE models crash on ROCm during initialization.
RUN git init /workspace/sglang && \
    cd /workspace/sglang && \
    git remote add origin https://github.com/sgl-project/sglang.git && \
    git fetch --depth 2 origin 1953efb60e82250cd704199833dcaf5075928f29 && \
    git checkout 1953efb60e82250cd704199833dcaf5075928f29~1 && \
    /opt/venv/bin/pip install -e ".[all]" 2>&1 | tail -5 && \
    git config user.email "build@eval" && git config user.name "eval" && \
    git add -A && \
    git checkout --orphan _baseline && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; \
    git branch -M _baseline main && \
    git reflog expire --expire=now --all && \
    git remote remove origin 2>/dev/null; \
    git gc --prune=now -q

# Replace base image's sglang with the same pre-fix checkout to prevent
# the agent from reading the fix from /sgl-workspace/sglang/.
# The base image (v0.5.9-rocm720-mi35x-20260226) contains post-fix code
# including CompressedTensorsWNA16TritonMoE.
RUN rm -rf /sgl-workspace/sglang && \
    cp -a /workspace/sglang /sgl-workspace/sglang && \
    rm -rf /sgl-workspace/sglang/.git && \
    cd /sgl-workspace/sglang && \
    git init && git add -A && git commit -m "base" 2>/dev/null || true

# The base image's editable install points to /sgl-workspace/sglang.
# Since the copy above replaced it with pre-fix code, edits there take
# effect at runtime. No second pip install needed.

# Add Arist12 fork remotes for result delivery (SSH keys mounted at runtime)
RUN cd /sgl-workspace/sglang && git remote add fork git@github.com:Arist12/sglang.git 2>/dev/null || true
RUN cd /sgl-workspace/aiter && git remote add fork git@github.com:Arist12/aiter.git 2>/dev/null || true

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

COPY bench_kimi_w4a16.sh /workspace/bench_kimi_w4a16.sh
RUN chmod +x /workspace/bench_kimi_w4a16.sh

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
