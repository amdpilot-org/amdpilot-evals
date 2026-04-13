ARG BASE_IMAGE=rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226
FROM ${BASE_IMAGE}
WORKDIR /workspace
RUN git clone --depth 100 https://github.com/sgl-project/sglang.git /workspace/sglang && \
    cd /workspace/sglang && git checkout e835601fb72ef2cbb53b2c6e2b00fe3370d3020e~1
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
