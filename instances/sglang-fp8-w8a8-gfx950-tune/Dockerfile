FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/sglang && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin c0a4408f780bc60be13c78b8217fb6b8899a4b73 && \
    git checkout -f c0a4408f780bc60be13c78b8217fb6b8899a4b73~1 && \
    /opt/venv/bin/pip install --no-deps -e python/
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/sglang /workspace/sglang
CMD ["sleep", "infinity"]
