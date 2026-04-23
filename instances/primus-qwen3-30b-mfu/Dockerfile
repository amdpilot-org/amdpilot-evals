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

# Replace the original detect_interface.sh (which depends on `ip`,
# unavailable in this slim base) with a /proc-based detector. Works
# in any container without iproute2.
RUN cat > /workspace/detect_interface.sh <<'EOF' && chmod +x /workspace/detect_interface.sh
#!/bin/bash
# Find the interface that handles the default route (no `ip` dependency).
IFACE=$(awk '$2 == "00000000" { print $1; exit }' /proc/net/route 2>/dev/null)
if [ -z "$IFACE" ]; then
    # Fallback: first non-loopback / non-docker interface from /sys.
    for cand in $(ls /sys/class/net/); do
        case "$cand" in
            lo|docker*|veth*|br-*|tailscale*|usb*) continue ;;
        esac
        if [ -d "/sys/class/net/$cand" ]; then IFACE="$cand"; break; fi
    done
fi
if [ -n "$IFACE" ]; then
    echo "export GLOO_SOCKET_IFNAME=$IFACE"
    echo "export NCCL_SOCKET_IFNAME=$IFACE"
fi
EOF

# gfx950 (MI355X) torch HIP runtime needs these set for kernel
# dispatch. Setting them here too (not just task.yaml env) so bare
# `docker run` works for ad-hoc smoke tests.
ENV PYTORCH_ROCM_ARCH=gfx950 \
    AITER_ROCM_ARCH="gfx942;gfx950" \
    HSA_NO_SCRATCH_RECLAIM=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib: \
    PATH=/opt/rocm/llvm/bin:/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Seed bench_config.env with PATH so verifications source it cleanly.
# bench_mfu.sh auto-detects GLOO/NCCL socket interfaces at runtime.
RUN echo 'export PATH="/opt/venv/bin:$PATH"' > /workspace/bench_config.env

CMD ["sleep", "infinity"]
