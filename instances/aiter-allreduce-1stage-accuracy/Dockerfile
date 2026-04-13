FROM lmsysorg/sglang-rocm:v0.5.9-rocm720-mi35x-20260326
WORKDIR /workspace

# Clean up git to prevent the executor from finding the fix via history.
RUN cd /sgl-workspace/aiter && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/aiter /workspace/aiter

CMD ["sleep", "infinity"]
