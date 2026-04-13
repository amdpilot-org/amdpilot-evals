# Kernel Launches on Wrong HIP Stream Causing Race Conditions

## Problem

Some GPU kernel launches in the compute library produce incorrect or non-deterministic results when called from a non-default HIP/CUDA stream. This causes race conditions in multi-stream scheduling and overlapped execution.

Symptoms include intermittent incorrect sampling outputs, non-deterministic results, and occasional data corruption when multiple operations are scheduled on different streams. The issue is particularly visible during speculative decoding or overlapped prefill/decode workloads where stream ordering is critical.

## Reproduction

The bug manifests when kernels are called from a non-default stream:

```python
# Pseudocode:
# 1. Create a non-default CUDA/HIP stream
# 2. On that stream, prepare input data
# 3. Call a kernel from the compute library
# 4. The output may be incorrect or non-deterministic
```

## Environment

- GPU compute libraries at `/sgl-workspace/aiter`
- Docker container with ROCm 7.2, PyTorch, AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands
- Single GPU sufficient for testing

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness runs sampling and kernel operations on non-default streams and checks for correctness against reference outputs computed on the default stream. A score of 100.0 means all stream-aware operations produce correct results.

**Important**: The fix requires modifying C/C++ source files. After making changes, clear JIT caches under `aiter/jit/` and recompile for changes to take effect.
