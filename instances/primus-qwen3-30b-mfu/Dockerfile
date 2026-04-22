# Primus Qwen3-30B-A3B MoE pretraining MFU eval (Issue #1).
#
# Base image is the pre-built Primus + Primus-Turbo image (already
# patched, ROCm 7.0, triton 3.4.0). amdpilot will pull from this
# registry tag if the image is not already present locally.
FROM ghcr.io/amdpilot-org/primus-mi355x-ready:v1

WORKDIR /workspace

# Overlay the canonical bench wrapper, test harness, and task description.
# Identical content lives in the base image already; the COPYs ensure that
# rebuilds always pick up the in-repo source of truth.
COPY bench_mfu.sh /workspace/bench_mfu.sh
RUN chmod +x /workspace/bench_mfu.sh

COPY test_harness.py /workspace/test_harness.py
COPY task_description.md /workspace/task_description.md

CMD ["sleep", "infinity"]
