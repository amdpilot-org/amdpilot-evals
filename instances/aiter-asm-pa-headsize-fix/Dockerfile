FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/aiter && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin e27a1807d67b731e0cc714a474776cf63bc6e30c && \
    git checkout -f e27a1807d67b731e0cc714a474776cf63bc6e30c~1 && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/aiter /workspace/aiter
CMD ["sleep", "infinity"]
