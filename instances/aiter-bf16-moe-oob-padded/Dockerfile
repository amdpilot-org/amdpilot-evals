FROM lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260406
WORKDIR /workspace
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
RUN ln -sf /sgl-workspace/aiter /workspace/aiter
CMD ["sleep", "infinity"]
