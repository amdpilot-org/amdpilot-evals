# MoE Intermediate Buffer Undersized When splitK > 1

## Context

The `ck_moe_stage1()` function in the fused MoE module allocates an intermediate output buffer (`tmp_out`) with shape `(token_num, topk, D)` based on the original token count. However, when `splitK > 1`, the sorted token index list is padded to be larger than `token_num * topk`. The CK kernel iterates over this padded sorted index list and writes results into `tmp_out`, but the buffer is too small to hold all the entries, causing the kernel to write past the buffer bounds.

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
