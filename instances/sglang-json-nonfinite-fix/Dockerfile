FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/sglang && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin && \
    git checkout -f e4d06b3db23d77ea21aebc14a5d398a81041e7c0~1 && \
    /opt/venv/bin/pip install --no-deps -e python/
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
