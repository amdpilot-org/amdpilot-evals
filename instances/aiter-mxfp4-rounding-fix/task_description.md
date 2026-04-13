# MXFP4 Quantization Produces Incorrect Values

## Context

The MXFP4 (Microscaling FP4) quantization in AITER produces incorrect quantized values for certain input ranges. The quantized output does not match the expected FP4 E2M1 specification for some boundary and denormal values.

This affects any workload using MXFP4 quantization through AITER, including FP4 MoE GEMM operations.

## How to reproduce

Quantize tensors containing values near FP4 representable boundaries and compare the output against the FP4 E2M1 specification. Some values will be rounded incorrectly.

## Environment

- AITER at `/sgl-workspace/aiter`
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
