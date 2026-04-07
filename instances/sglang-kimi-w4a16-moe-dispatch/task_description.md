# Fix W4A16-Quantized MoE Model Loading on ROCm

## Context

Kimi K2.5 is a large MoE model (DeepseekV3 architecture, 384 experts) that uses INT4
weight-only quantization (W4A16: 4-bit weights, 16-bit activations) via the
`compressed-tensors` format.

When loading this model on ROCm/HIP (AMD MI300X), the MoE layer fails to
initialize. The error indicates missing or incompatible kernel support for the
quantized MoE dispatch path on this platform.

The model loads and serves correctly on NVIDIA GPUs but crashes on AMD GPUs
during MoE layer initialization.

## Task

Fix the quantized MoE dispatch so that W4A16 MoE models can load and run
correctly on ROCm/HIP. The fix should:

1. Allow the MoE layer to initialize without errors on AMD GPUs
2. Correctly handle the packed INT4 weight format (int32-packed -> working compute format)
3. Produce numerically correct MoE output (verify against a reference)
4. Use a kernel backend that is available on ROCm

## Reproduction

The failure occurs during MoE layer initialization when loading a W4A16-quantized
model on ROCm. The method selection path in the compressed-tensors quantization
code picks a backend that is not available on HIP.

The test harness provides a synthetic reproduction that exercises the dispatch
and forward pass without requiring full model weights.

## Environment

- Repository: sgl-project/sglang (code at `/workspace/sglang`)
- Docker container with ROCm, PyTorch, AMD GPU (MI300X)
- Use `/opt/venv/bin/python3` for all commands
- The MoE quantization dispatch is in the compressed-tensors quantization module

## Verification

```bash
cd /workspace/sglang && /opt/venv/bin/python3 /workspace/test_harness.py
```
