FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/sglang && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin 93422f27d6f41c2cfb0315c9569c98fc3a55a322 && \
    git checkout -f 93422f27d6f41c2cfb0315c9569c98fc3a55a322~1 && \
    /opt/venv/bin/pip install --no-deps -e python/
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/sglang /workspace/sglang
CMD ["sleep", "infinity"]
