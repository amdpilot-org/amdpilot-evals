ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}
WORKDIR /workspace

# Check out sglang at pre-fix state (parent of PR #17863 merge commit)
RUN git init /workspace/sglang && \
    cd /workspace/sglang && \
    git remote add origin https://github.com/sgl-project/sglang.git && \
    git fetch --depth 2 origin 1953efb60e82250cd704199833dcaf5075928f29 && \
    git checkout 1953efb60e82250cd704199833dcaf5075928f29~1 && \
    # Install sglang in the pre-fix state \
    /opt/venv/bin/pip install -e ".[all]" 2>&1 | tail -5 && \
    # Scrub git history to prevent solution leak \
    git config user.email "build@eval" && git config user.name "eval" && \
    git add -A && \
    git checkout --orphan _baseline && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; \
    git branch -M _baseline main && \
    git reflog expire --expire=now --all && \
    git remote remove origin 2>/dev/null; \
    git gc --prune=now -q

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
