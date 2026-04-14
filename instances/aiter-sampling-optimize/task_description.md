# Optimize TopK/TopP Sampling Kernel

The TopK/TopP sampling kernel in AITER has excessive latency. Profiling shows the kernel spends significant time in synchronization barriers within the sampling loop.

For batch_size=1 and vocab_size=128256, the kernel takes ~0.75ms which is too slow for interactive serving. The target is to reduce this below 0.5ms.

Locate the sampling kernel source in the AITER codebase and the
corresponding sampling test suite.

After making changes, rebuild. **Important**: AITER sampling kernels use
template-based JIT compilation with cached build artifacts. Editing `.cuh`
headers does NOT automatically invalidate the cache. You MUST delete the
relevant cached build artifacts before running `setup.py develop`, or your
changes will have no effect.

## Environment

- AITER at `/sgl-workspace/aiter`
- Use `/opt/venv/bin/python3`
- AMD MI355X GPU (gfx950)

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
