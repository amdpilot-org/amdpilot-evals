# ROCm MXFP4 MoE: AITER fused MoE fails at certain tensor-parallel sizes

On ROCm with AITER fused MoE enabled, serving some MXFP4 MoE models (for example `amd/MiniMax-M2.1-MXFP4`) with `tensor_parallel_size=4` causes workers to crash inside the AITER fused MoE / `device_gemm` path with a runtime error indicating the GEMM problem is not supported. The same model may load successfully with a smaller tensor parallel size (for example TP=2) while TP=4 fails.

The failure appears when combining MXFP4-quantized MoE weights, ROCm AITER MoE, and a TP split that yields an incompatible effective expert intermediate width after partitioning.

## Environment

- vLLM source at `/workspace/vllm`
- Python: `/opt/venv/bin/python3`
- ROCm / AMD GPU stack as in the container image

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
