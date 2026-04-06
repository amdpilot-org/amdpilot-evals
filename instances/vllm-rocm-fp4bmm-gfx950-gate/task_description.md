# Bug: Default VLLM_ROCM_USE_AITER_FP4BMM=True crashes on MI300X (gfx942)

## Symptom

vLLM crashes on AMD MI300X/MI300A (gfx942) GPUs with the default configuration because `VLLM_ROCM_USE_AITER_FP4BMM` defaults to `True`. MI300X uses CDNA3 architecture which does **not** support FP4 (MXFP4) instructions -- those are CDNA4-only (gfx950: MI325X/MI350X/MI355X).

When vLLM attempts FP4 block matrix multiply on MI300X, it crashes with:
```
RuntimeError: MXFP4 quantization is not supported on gfx942
```

This affects ~90% of AMD GPU users since MI300X is the most common AMD data center GPU.

## Root Cause

Two functions in `vllm/_aiter_ops.py` check environment variables but never verify hardware capability:

1. `is_fp4bmm_enabled()` -- returns `True` on MI300X when `VLLM_ROCM_USE_AITER_FP4BMM=True` (the default), even though the GPU cannot execute FP4 instructions.

2. `is_asm_fp4_gemm_dynamic_quant_enabled()` -- same pattern, missing hardware check.

The AITER library itself knows FP4 only works on gfx950, but vLLM never queries this hardware capability before enabling the FP4 code paths.

## Affected files

- `vllm/_aiter_ops.py` -- the two functions listed above need a hardware capability guard

## Expected behavior

- `is_fp4bmm_enabled()` should return `False` on gfx942 (MI300X) regardless of the env var setting
- `is_asm_fp4_gemm_dynamic_quant_enabled()` should return `False` on gfx942 regardless of the env var setting
- Both functions should only return `True` on gfx950 (MI325X/MI350X/MI355X) when the env var is also enabled
- The fix should use the existing `on_gfx950()` function from `vllm/platforms/rocm.py` which is already used elsewhere in the codebase for gfx950-only feature gating

## Workaround (for reference only)

```bash
export VLLM_ROCM_USE_AITER_FP4BMM=0
```
