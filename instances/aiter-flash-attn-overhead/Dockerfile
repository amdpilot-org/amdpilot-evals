ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}

WORKDIR /workspace

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

RUN ln -sf /sgl-workspace/aiter /workspace/aiter

CMD ["sleep", "infinity"]
