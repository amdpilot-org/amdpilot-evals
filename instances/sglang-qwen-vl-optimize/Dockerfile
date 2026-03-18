FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317

WORKDIR /workspace

# Clone user's sglang fork to a non-conflicting path (avoid shadowing the
# installed sglang package — /workspace/sglang/ would be resolved as a
# namespace package by Python).
RUN git clone --depth 100 https://github.com/Arist12/sglang.git /workspace/sglang-fork && \
    cd /workspace/sglang-fork && \
    git config user.email "amdpilot@amd.com" && \
    git config user.name "amdpilot"

# sglang.bench_serving imports sglang.benchmark.datasets which is excluded
# from the pip install.  Add the repo's python/ dir to PYTHONPATH so the
# benchmark subpackage is importable alongside the installed runtime.
ENV PYTHONPATH="/sgl-workspace/sglang/python:${PYTHONPATH}"

# bench_serving image dataset needs pillow
RUN /opt/venv/bin/pip install --no-cache-dir pillow 2>/dev/null || true

COPY bench_qwen_vl.sh /workspace/bench_qwen_vl.sh
RUN chmod +x /workspace/bench_qwen_vl.sh

COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
