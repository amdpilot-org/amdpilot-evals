# Optimize TopK/TopP Sampling Kernel

The TopK/TopP sampling kernel in AITER (`csrc/cpp_itfs/sampling/sampling.cuh`) has excessive latency. Profiling shows the kernel spends significant time in synchronization barriers within the sampling loop.

For batch_size=1 and vocab_size=128256, the kernel takes ~0.75ms which is too slow for interactive serving. The target is to reduce this to under 0.5ms.

The kernel is at: `/sgl-workspace/aiter/csrc/cpp_itfs/sampling/sampling.cuh`
Tests are at: `/sgl-workspace/aiter/op_tests/test_sampling.py`

After making changes, rebuild:
```bash
cd /sgl-workspace/aiter && \
  rm -rf /root/.aiter/build/top_k_top_p_sampling_from_probs* && \
  /opt/venv/bin/python3 setup.py develop
```
**Important**: AITER sampling kernels use template-based JIT compilation
cached in `/root/.aiter/build/`. Editing `.cuh` headers does NOT
automatically invalidate the cache. You MUST delete the cached build
directory before running the test, or your changes will have no effect.

## Environment

- AITER at `/sgl-workspace/aiter`
- Use `/opt/venv/bin/python3`
- AMD MI355X GPU (gfx950)

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
