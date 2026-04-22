# Primus Qwen3-30B-A3B MoE pretraining MFU eval (Issue #1).
#
# Base image is the locally-built primus-mi355x-flat:v1, which already
# has /workspace/primus_train/Primus + Primus-Turbo + the triton 3.4.0
# patches applied. We layer two things on top:
#
#  1. The canonical bench wrapper / test harness / task description.
#  2. uv (the Python package manager) at /root/.local/bin so the
#     kimi-cli runtime can `source $HOME/.local/bin/env` on container
#     start — without this the executor hits "No such file or
#     directory" and trials exit 137 immediately.
FROM primus-mi355x-flat:v1

WORKDIR /workspace

# Install uv with its activation script so kimi-cli wheel install works.
RUN mkdir -p /tmp && \
    curl -LsSf https://astral.sh/uv/install.sh | TMPDIR=/tmp sh && \
    test -x /root/.local/bin/uv && \
    test -f /root/.local/bin/env

# Overlay the canonical bench wrapper, test harness, and task description.
COPY bench_mfu.sh /workspace/bench_mfu.sh
RUN chmod +x /workspace/bench_mfu.sh

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
