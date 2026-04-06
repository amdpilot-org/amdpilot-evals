# MoE Intermediate Buffer Undersized When splitK > 1

## Context

The fused MoE kernel allocates an intermediate output buffer based on the original token count. However, when `splitK > 1`, the sorted token index list is padded to be larger than the original size. The kernel iterates over this padded list and writes results into the buffer, but the buffer is too small to hold all the entries, causing out-of-bounds writes.

This results in either a crash or silently incorrect MoE output during inference when the splitK configuration is greater than 1.

## Environment

- AITER at `/sgl-workspace/aiter`
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
