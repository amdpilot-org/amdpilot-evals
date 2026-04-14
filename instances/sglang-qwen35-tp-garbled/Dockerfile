FROM rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260406

WORKDIR /workspace

# The base image already has the buggy aiter and sglang at commit 0906e45ce
# (matching the issue environment). Clean git history to prevent the agent
# from finding the fix via remote browsing.
RUN cd /sgl-workspace/aiter && \
    git remote remove origin 2>/dev/null; \
    git tag -d $(git tag -l) 2>/dev/null; \
    git branch -D main 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now --aggressive -q
RUN cd /sgl-workspace/sglang && \
    git remote remove origin 2>/dev/null; \
    git tag -d $(git tag -l) 2>/dev/null; \
    git branch -D main 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now --aggressive -q

# Install httpx for the test script (matches the issue's reproduction)
RUN /opt/venv/bin/pip install httpx 2>/dev/null || true

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

# Symlink repos for agent access. Note: the aiter repo root contains an
# "aiter/" subdirectory which is the Python package; naming the symlink
# "aiter" would shadow the installed package on sys.path (/workspace is
# on sys.path because WORKDIR=/workspace). Use "aiter_repo" to avoid this.
RUN ln -sf /sgl-workspace/sglang /workspace/sglang && \
    ln -sf /sgl-workspace/aiter /workspace/aiter_repo

CMD ["sleep", "infinity"]
