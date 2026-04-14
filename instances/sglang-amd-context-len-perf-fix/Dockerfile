ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && \
    git fetch --depth 100 origin 67c295b5f5a97cd5742aaf453a7cba262c8ba127 && \
    git checkout -f 67c295b5f5a97cd5742aaf453a7cba262c8ba127~1 && \
    git remote remove origin 2>/dev/null; \
    git reflog expire --expire=now --all && \
    git gc --prune=now -q
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
