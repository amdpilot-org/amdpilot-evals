FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326
WORKDIR /workspace

# The base image's aiter contains the pre-fix version.  No git checkout needed.
# Clean up git to prevent the executor from finding the fix via history.
RUN cd /sgl-workspace/aiter && \
    git remote remove origin 2>/dev/null; \
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
