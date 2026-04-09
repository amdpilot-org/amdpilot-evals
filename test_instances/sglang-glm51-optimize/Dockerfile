FROM rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260406

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/sgl-project/sglang.git /workspace/sglang

# Update installed sglang and aiter runtimes to latest upstream (upstreamable results)
RUN cd /sgl-workspace/sglang && git pull origin main && \
    /opt/venv/bin/pip install --no-deps -e "python[all]" 2>/dev/null || true
RUN cd /sgl-workspace/aiter && git pull origin main && \
    /opt/venv/bin/pip install --no-deps -e . 2>/dev/null || true

# Add Arist12 fork remotes for result delivery (SSH keys mounted at runtime)
RUN cd /sgl-workspace/sglang && git remote add fork git@github.com:Arist12/sglang.git 2>/dev/null || true
RUN cd /sgl-workspace/aiter && git remote add fork git@github.com:Arist12/aiter.git 2>/dev/null || true

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

# GLM-5.1 (glm_moe_dsa architecture) requires recent transformers; upgrade
RUN /opt/venv/bin/pip install --upgrade transformers huggingface_hub 2>/dev/null || true

COPY bench_glm51.sh /workspace/bench_glm51.sh
RUN chmod +x /workspace/bench_glm51.sh

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
