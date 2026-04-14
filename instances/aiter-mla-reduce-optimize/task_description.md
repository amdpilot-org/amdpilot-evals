# Optimize MLA Reduce Kernel

The Multi-Latent Attention (MLA) reduce kernel in AITER has high latency that impacts end-to-end inference performance for models using MLA (e.g., DeepSeek variants).

Profiling with a sparse MLA workload (`batch=1, context=4000, nhead=16,2, dtype=bf16, kvd=bf16`) shows the reduce kernel takes approximately 18.2 microseconds. The target is to reduce this to under 13 microseconds (at least 25% improvement).

The kernel performs reduction of partial attention outputs across splits. Key bottlenecks include:
- Suboptimal handling of different split counts (small vs. large)
- Memory access patterns in the reduce phase
- Synchronization overhead between compute and memory operations

Locate the MLA reduce kernel source in the AITER codebase.

After making changes, rebuild the kernel. **Important**: AITER uses JIT
compilation. Editing `.cu` files does NOT automatically rebuild the cached
`.so` module. You MUST delete the relevant cached module under
`aiter/jit/` before running `setup.py develop`, or your changes will have
no effect.

## Environment

- AITER at `/sgl-workspace/aiter`
- Use `/opt/venv/bin/python3`
- AMD MI355X GPU (gfx950)

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
