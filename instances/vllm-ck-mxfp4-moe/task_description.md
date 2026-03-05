# Add CK Backend for MXFP4 MoE Quantization on ROCm

## Problem

vLLM's MXFP4 quantization (`mxfp4.py`) has no fused MoE support on ROCm. Models using MXFP4 quantization with MoE layers cannot run efficiently — they fall back to dense computation or fail entirely on AMD GPUs.

## Affected Files

- `vllm/_aiter_ops.py`
- `vllm/model_executor/layers/quantization/mxfp4.py`

## Environment

- Use `/opt/venv/bin/python3` for all commands

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
