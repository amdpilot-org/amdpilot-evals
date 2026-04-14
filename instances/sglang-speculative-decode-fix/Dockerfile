FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326
WORKDIR /workspace

# Re-introduce the stream sync bug in aiter's CK-tile attention kernels.
# The fix was replacing stream_config{nullptr} with stream_config{stream} —
# using nullptr causes the kernel to run on the default HIP stream instead
# of the caller's stream, causing race conditions and garbled output.
# Bug location: csrc/py_itfs_ck/mha_fwd_kernels.cu:353,
#               csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu:580,
#               csrc/py_itfs_ck/mha_batch_prefill_kernels.cu:744
RUN cd /sgl-workspace/aiter && \
    sed -i 's/ck_tile::stream_config stream_config{stream}/ck_tile::stream_config stream_config{nullptr}/g' \
        csrc/py_itfs_ck/mha_fwd_kernels.cu \
        csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu \
        csrc/py_itfs_ck/mha_batch_prefill_kernels.cu && \
    rm -rf aiter/jit/build/ aiter/jit/*.so

# Clean up git to prevent the executor from finding the fix via history.
RUN cd /sgl-workspace/aiter && \
    git remote remove origin 2>/dev/null; \
    git add -A 2>/dev/null; \
    git commit -m "update" --allow-empty 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

# Remove flydsl: version check in aiter/ops/flydsl/__init__.py expects
# 0.1.1.dev409 but image has 0.0.1.dev95158637, crashing the import chain.
# The actual bug (CK-tile blockscale GEMM stream) is unrelated to flydsl.
RUN /opt/venv/bin/pip uninstall -y flydsl 2>/dev/null || true

# Ensure aiter is importable
ENV PYTHONPATH="/sgl-workspace/aiter:${PYTHONPATH}"

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

# Symlink for easy navigation
RUN ln -sf /sgl-workspace/aiter /workspace/aiter && \
    ln -sf /sgl-workspace/sglang /workspace/sglang 2>/dev/null; true

CMD ["sleep", "infinity"]
