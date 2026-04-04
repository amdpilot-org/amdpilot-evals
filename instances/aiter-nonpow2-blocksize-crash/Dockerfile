FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317
WORKDIR /workspace
RUN cd /sgl-workspace/aiter && \
    git checkout -f HEAD && \
    git fetch --depth 100 origin f812f276446c603df6358899ce769dc837f73b0e && \
    git checkout -f f812f276446c603df6358899ce769dc837f73b0e~1
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/aiter /workspace/aiter
CMD ["sleep", "infinity"]
