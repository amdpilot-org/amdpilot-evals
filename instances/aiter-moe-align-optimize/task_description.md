# Optimize moe_align_block_size Kernel

The `moe_align_block_size` kernel in AITER (`csrc/kernels/moe_align_block_size_kernels.cu`) is a bottleneck for MoE (Mixture of Experts) inference. The kernel aligns token counts to block boundaries for efficient expert-parallel execution.

The current implementation uses excessive shared memory, limiting performance especially for large expert counts (64+). The kernel needs to be optimized to:
1. Reduce shared memory usage
2. Improve throughput for E=64, tokens=4096, topk=8

The kernel is at: `/sgl-workspace/aiter/csrc/kernels/moe_align_block_size_kernels.cu`
The Python wrapper is at: `/sgl-workspace/aiter/aiter/ops/moe_op.py` (or similar)
Tests: `/sgl-workspace/aiter/op_tests/test_moe.py` (if exists)

After making changes, rebuild the kernel:
```bash
cd /sgl-workspace/aiter && \
  rm -rf aiter/jit/module_moe_asm.so aiter/jit/build/module_moe_asm && \
  /opt/venv/bin/python3 setup.py develop
```
**Important**: AITER uses JIT compilation. Editing `.cu` files does NOT
automatically rebuild the cached `.so` module. You MUST delete the cached
module (`rm -rf aiter/jit/module_moe_asm.so aiter/jit/build/module_moe_asm`)
before running the test, or your changes will have no effect.

## Environment

- AITER at `/sgl-workspace/aiter`
- Use `/opt/venv/bin/python3`
- AMD MI355X GPU (gfx950)

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
