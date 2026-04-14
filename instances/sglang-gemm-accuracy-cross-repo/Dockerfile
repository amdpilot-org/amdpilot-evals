## Inject a MoE output scaling bug in aiter's fused_moe dispatch.
## The bug multiplies MoE output by 0.01, effectively silencing expert
## contributions.  This causes degraded model outputs.
FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326

WORKDIR /workspace

# Copy and run the bug injection script
COPY inject_bug.py /tmp/inject_bug.py
RUN python3 /tmp/inject_bug.py && rm /tmp/inject_bug.py

# Create a clean git baseline so the agent can't see the injection.
# Must set user.email/name for commit, then clean all branches/tags/remotes.
RUN cd /sgl-workspace/aiter && \
    git config user.email "build@local" && \
    git config user.name "build" && \
    git checkout --orphan _clean && git add -A && \
    git commit -m "baseline" && \
    git branch -D main 2>/dev/null; \
    git branch -M _clean main && \
    git tag -l | xargs -r git tag -d 2>/dev/null; \
    git remote | xargs -rn1 git remote remove 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now --aggressive -q

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
