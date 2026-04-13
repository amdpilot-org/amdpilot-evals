ARG BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260406
FROM ${BASE_IMAGE}
WORKDIR /workspace

# Surgically remove the OOB bounds checks added by commit 7eaedeb36
# ("Ck moe hot fix") while preserving template API compatibility with the
# rest of the CK tree. This restores the buggy code paths:
# 1. gridwise_moe_gemm.hpp: remove `token_offset < problem.NumTokens` ternary,
#    always access weights unconditionally (causes OOB on sentinel rows)
# 2. threadwise_tensor_slice_transfer_v7r3_scatter.hpp: remove dst bounds check,
#    always pass true to scatter write (causes OOB writes on sentinel rows)
COPY revert_oob_fix.py /tmp/revert_oob_fix.py
RUN /opt/venv/bin/python3 /tmp/revert_oob_fix.py && rm /tmp/revert_oob_fix.py

# Clear JIT-compiled kernel cache so next run recompiles from pre-fix source.
RUN rm -rf /sgl-workspace/aiter/aiter/jit/build/module_moe_ck2stages* \
           /sgl-workspace/aiter/aiter/jit/build/module_moe_cktile2stages* \
           /sgl-workspace/aiter/aiter/jit/module_moe_ck2stages* \
           /sgl-workspace/aiter/aiter/jit/module_moe_cktile2stages*

# Strip git history from aiter/CK and sglang to prevent data leak.
# Re-init with a clean single commit so the agent can track their changes.
RUN cd /sgl-workspace/aiter && \
    rm -rf .git/modules 3rdparty/composable_kernel/.git && \
    rm -rf .git && \
    git init && \
    git config user.email "build@eval" && git config user.name "eval" && \
    git add -A -- aiter/ csrc/ setup.py pyproject.toml && \
    git add -A -- 3rdparty/composable_kernel/include/ && \
    git commit -m "base" --allow-empty 2>/dev/null || git commit --allow-empty -m "base" && \
    git gc --prune=now -q 2>/dev/null || true

RUN cd /sgl-workspace/sglang && \
    rm -rf .git && \
    git init && \
    git config user.email "build@eval" && git config user.name "eval" && \
    git add -A -- python/ && \
    git commit -m "base" 2>/dev/null || git commit --allow-empty -m "base" && \
    git gc --prune=now -q 2>/dev/null || true

# Add fork remotes for result delivery (SSH keys mounted at runtime).
RUN cd /sgl-workspace/aiter && git remote add fork git@github.com:Arist12/aiter.git 2>/dev/null || true
RUN cd /sgl-workspace/sglang && git remote add fork git@github.com:Arist12/sglang.git 2>/dev/null || true

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

COPY bench_bf16_moe.sh /workspace/bench_bf16_moe.sh
RUN chmod +x /workspace/bench_bf16_moe.sh

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
