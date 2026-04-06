# GEMM Config Cache Pollution Causes KeyError on Second Call

## Context

The `gemm_afp4wfp4` fused kernel crashes with a `KeyError` on its second invocation during sglang inference with `quark_w4a4_mxfp4` quantization. The first call to the kernel succeeds, but the second call fails because the GEMM configuration dictionary returned by `get_gemm_config()` has been corrupted between calls.

The function `get_gemm_config()` in `aiter/ops/triton/utils/gemm_config_utils.py` uses an LRU cache to store configuration dictionaries. The fused kernel receives this cached dict reference and mutates it in-place during execution. When `get_gemm_config()` is called again with the same arguments, the LRU cache returns the previously mutated (corrupted) dictionary, which is now missing or has altered keys that the kernel expects.

## Affected Files

- `aiter/ops/triton/utils/gemm_config_utils.py`

## Environment

- AITER at `/sgl-workspace/aiter`
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
