FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260311

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/sgl-project/sglang.git /workspace/sglang

# transformers 4.57.1 doesn't recognize glm_moe_dsa; upgrade to 4.57.x
# NOTE: transformers 5.x breaks sglang's vendored DeepseekVL2Config (dataclass field ordering)
RUN /opt/venv/bin/pip install 'transformers>=4.57.3,<5.0' huggingface_hub 2>/dev/null || true

# Patch sglang for GLM-5 (glm_moe_dsa) compatibility:
# - transformers <5.0 doesn't know glm_moe_dsa → add AutoConfig fallback
# - model_config.py accesses rope_scaling without guard → add getattr default
COPY patch_sglang_glm5.py /tmp/patch_sglang_glm5.py
RUN /opt/venv/bin/python3 /tmp/patch_sglang_glm5.py && rm /tmp/patch_sglang_glm5.py

COPY bench_glm5.sh /workspace/bench_glm5.sh
RUN chmod +x /workspace/bench_glm5.sh

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
