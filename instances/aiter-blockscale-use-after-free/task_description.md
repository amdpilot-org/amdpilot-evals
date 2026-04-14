# FP8 Blockscale GEMM Produces Incorrect Results Under Memory Pressure

## Problem

The FP8 blockscale GEMM kernel (a8w8 blockscale variant using CK-tile) produces incorrect results when the GPU memory allocator is under pressure. In isolation or with light memory usage, the kernel produces correct outputs. However, during inference workloads that allocate and free many tensors (such as serving frameworks processing multiple requests), the GEMM output can have large errors (max absolute error ~21000 vs expected ~0.000008).

The corruption is intermittent and depends on memory allocation patterns — it only manifests when the GPU memory allocator recycles a specific freed memory block.

## Reproduction

The bug can be triggered by:
1. Running the blockscale GEMM kernel
2. Creating memory pressure by allocating and freeing tensors around the GEMM call
3. Checking the output against a reference computation

```python
# Pseudocode:
# 1. Allocate input tensors (A: int8, B: int8, x_scale, w_scale)
# 2. Create memory pressure (alloc/free cycle)
# 3. Run blockscale GEMM
# 4. Compare output to torch reference matmul
# 5. The error may be very large if freed memory was recycled
```

The issue is in the C++ kernel launcher code for the blockscale GEMM variant.

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

The harness runs blockscale GEMM operations under varying memory pressure conditions and checks output correctness. A score of 100.0 means all GEMM outputs are numerically correct.

**Important**: The fix requires modifying C/C++ source code. After making changes, clear JIT caches under `aiter/jit/` and recompile for changes to take effect.
