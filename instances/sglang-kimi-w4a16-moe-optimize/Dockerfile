ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260406
FROM ${BASE_IMAGE}

# Use /root as workdir to avoid aiter namespace package import conflict
WORKDIR /root

# Strip git history from sglang and aiter to prevent data leaks
RUN cd /sgl-workspace/sglang && \
    git config user.email "build@eval" && git config user.name "eval" && \
    git checkout --orphan _clean && \
    git add -A && git commit -m "baseline" && \
    git branch -D main 2>/dev/null; \
    git branch -M _clean main && \
    git reflog expire --expire=now --all && \
    git remote remove origin 2>/dev/null; \
    git gc --prune=now -q

RUN cd /sgl-workspace/aiter && \
    git config user.email "build@eval" && git config user.name "eval" && \
    git checkout --orphan _clean && \
    git add -A && git commit -m "baseline" && \
    git branch -D main 2>/dev/null; \
    git branch -M _clean main && \
    git reflog expire --expire=now --all && \
    git remote remove origin 2>/dev/null; \
    git gc --prune=now -q

# Add Arist12 fork remotes for result delivery (SSH keys mounted at runtime)
RUN cd /sgl-workspace/sglang && git remote add fork git@github.com:Arist12/sglang.git 2>/dev/null || true
RUN cd /sgl-workspace/aiter && git remote add fork git@github.com:Arist12/aiter.git 2>/dev/null || true

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

COPY bench_kimi_w4a16.sh /workspace/bench_kimi_w4a16.sh
COPY bench_kimi_w4a16_smoke.sh /workspace/bench_kimi_w4a16_smoke.sh
RUN chmod +x /workspace/bench_kimi_w4a16.sh /workspace/bench_kimi_w4a16_smoke.sh

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
