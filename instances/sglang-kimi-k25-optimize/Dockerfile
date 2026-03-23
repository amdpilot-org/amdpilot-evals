FROM rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/sgl-project/sglang.git /workspace/sglang

# Kimi-K2.5 needs trust-remote-code; ensure transformers/huggingface_hub are fresh
RUN /opt/venv/bin/pip install --upgrade transformers huggingface_hub 2>/dev/null || true

# Add Arist12 fork remotes for result delivery (SSH keys mounted at runtime)
RUN cd /sgl-workspace/sglang && git remote add fork git@github.com:Arist12/sglang.git 2>/dev/null || true
RUN cd /sgl-workspace/aiter && git remote add fork git@github.com:Arist12/aiter.git 2>/dev/null || true

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

COPY bench_kimi_k25.sh /workspace/bench_kimi_k25.sh
RUN chmod +x /workspace/bench_kimi_k25.sh

COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
