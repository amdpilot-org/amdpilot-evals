FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260311

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/sgl-project/sglang.git /workspace/sglang

# transformers 4.57.1 doesn't recognize glm_moe_dsa; upgrade with deps
RUN /opt/venv/bin/pip install --upgrade transformers huggingface_hub 2>/dev/null || true

COPY bench_glm5.sh /workspace/bench_glm5.sh
RUN chmod +x /workspace/bench_glm5.sh

COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
