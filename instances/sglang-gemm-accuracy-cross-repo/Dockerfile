## Surgical revert: only re-introduce the use-after-free bug in the
## blockscale GEMM kernel, keeping all other aiter code at HEAD.
FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326

WORKDIR /workspace

# Revert ONLY the single GEMM header file to its pre-fix state.
# The fix (3e4552bcc) moved x_scale_t to function scope; we undo that
# so the tensor is destroyed while the async kernel is still reading it.
RUN cd /sgl-workspace/aiter && \
    git checkout 3e4552bcc005911ad438de8b09ec5ecc84c03dc7~1 -- \
        csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_cktile_common.cuh && \
    rm -rf jit/module_gemm_a8w8_blockscale* build/ && \
    git checkout --orphan _clean && git add -A && \
    git commit -m "baseline with GEMM bug" && \
    git branch -D main 2>/dev/null; git branch -M _clean main && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/aiter /workspace/aiter

CMD ["sleep", "infinity"]
