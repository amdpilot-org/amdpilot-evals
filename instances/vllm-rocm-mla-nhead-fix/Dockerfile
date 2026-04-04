FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace
RUN mkdir -p /opt/venv/bin && ln -sf /usr/bin/python3 /opt/venv/bin/python3
RUN git clone --depth 100 https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && \
    git fetch --depth 100 origin 1b117cb0ac51b84dd7aed364e6e802dea5147ca6 && \
    git checkout -f 1b117cb0ac51b84dd7aed364e6e802dea5147ca6~1 && \
    pip install --no-build-isolation --no-deps -e .
COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
