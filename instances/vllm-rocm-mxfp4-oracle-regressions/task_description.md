# MXFP4-quantized MoE models crash or produce incorrect results on ROCm

## Symptom

Models using MXFP4 quantization with Mixture-of-Experts layers crash or produce incorrect results on ROCm GPUs. The failures manifest in several ways depending on the model configuration and GPU architecture:

- Runtime crashes with reshape errors or kernel launch failures for certain model dimension sizes
- Incorrect numerical output when MoE expert computations use wrong padding or alignment
- Crashes on some GPU architectures where the selected backend is not actually supported

The issue affects multiple model configurations and is not limited to a single failure mode.

## How to reproduce

Run an MXFP4-quantized MoE model on a ROCm GPU. Different model configurations may trigger different failure modes. Models with non-standard intermediate sizes are more likely to crash.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
