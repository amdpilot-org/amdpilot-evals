# FP8 GEMM Kernel Produces Incorrect Results Under Multi-Stream Workloads

## Problem

An FP8 GEMM kernel in the aiter library produces incorrect results when the caller operates on a non-default HIP stream. A single-stream (default stream) caller sees correct results.

This causes downstream issues in inference frameworks that use non-default streams for concurrent execution (e.g., speculative decoding with overlap scheduling).

## Reproduction

The kernel works correctly when called on the default HIP stream, but produces incorrect results when the caller uses a non-default stream in a concurrent multi-stream workload. The kernel does not respect the caller's execution context.

## Environment

- AITER at `/sgl-workspace/aiter`
- Docker container with ROCm, PyTorch, AMD MI355X GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

**Important**: If you modify C/C++ source files, you may need to delete cached JIT-compiled modules under `aiter/jit/` and rebuild for changes to take effect.
