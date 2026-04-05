FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/aiter && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin 357e60096b08291886b89c354a97f8fb20a64728 && \
    git checkout -f 357e60096b08291886b89c354a97f8fb20a64728~1 && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/aiter /workspace/aiter
CMD ["sleep", "infinity"]
