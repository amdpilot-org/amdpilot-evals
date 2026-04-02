FROM rocm/vllm-dev:rocm7.2.1_v0.18.0_20260330
WORKDIR /workspace

# Pin to the commit immediately before the known stride fix (#37228) to avoid
# "already fixed upstream" contamination in the eval bundle.
RUN git clone https://github.com/vllm-project/vllm.git /workspace/vllm && \
    cd /workspace/vllm && git checkout a0e8c74005f5782fad30be912c12cd37fc2813e9

# Force runtime to use the checked-out source, not the pre-installed wheel.
RUN cd /workspace/vllm && \
    pip uninstall -y vllm || true && \
    pip install -e . --no-deps --no-build-isolation

ENV PYTHONPATH=/workspace/vllm

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md
CMD ["sleep", "infinity"]
