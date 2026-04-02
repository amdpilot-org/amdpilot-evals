FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace

# Pin to a pre-fix commit / known problematic range to avoid silently using a
# later upstream behavior change.
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && git checkout 2647163674720242cee78dae3f7c7b98539ed029

# Force runtime to use the checked-out source, not the pre-installed wheel.
RUN cd /workspace/vllm && \
    pip uninstall -y vllm || true && \
    pip install -e . --no-deps --no-build-isolation

ENV PYTHONPATH=/workspace/vllm

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
