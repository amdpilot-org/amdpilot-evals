# BF16 MoE Kernel GPU Memory Access Fault

## Problem

When running large BF16 MoE models (e.g., Qwen3-Next-80B-A3B-Instruct at TP=8) using the AITER CK 2-stage MoE kernel, the system crashes with a GPU memory access fault:

```
Memory access fault by GPU node-3 on address 0x7d4607000000.
Reason: Write access to a read-only page.
```

The crash occurs during the MoE GEMM stage-1 kernel execution, specifically when processing batches where the number of tokens routed to an expert does not evenly fill the kernel's processing blocks.

## Task

Investigate and fix the out-of-bounds memory access in the BF16 CK 2-stage MoE kernel path. The fix should be in the Python-level MoE code at `/workspace/aiter/aiter/fused_moe.py`. The kernel C++ source cannot be easily modified.

## Environment

- AITER at `/sgl-workspace/aiter` (also symlinked at `/workspace/aiter`)
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
