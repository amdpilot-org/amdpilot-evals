# MXFP4 Quantization Produces Incorrect Rounding in fp4_utils.py

## Context

The MXFP4 (Microscaling FP4) quantization utility in `aiter/utility/fp4_utils.py` has a rounding bug that causes incorrect quantized values. The rounding logic does not properly handle the FP4 format's limited mantissa bits, leading to values that should round up being truncated instead.

This affects any workload using MXFP4 quantization through AITER, including FP4 MoE GEMM operations.

## Affected Files

- `aiter/utility/fp4_utils.py`

## Environment

- AITER at `/sgl-workspace/aiter`
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
