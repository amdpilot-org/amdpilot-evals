FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/sglang && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin 7c7b2a8c97816b84c6c90ebe28e7e1a9ea334888 && \
    git checkout -f 7c7b2a8c97816b84c6c90ebe28e7e1a9ea334888~1 && \
    /opt/venv/bin/pip install --no-deps -e python/ && \
    rm -f .git/FETCH_HEAD && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/sglang /workspace/sglang
CMD ["sleep", "infinity"]
