# vLLM MXFP4 quantization: MoE models run slowly on ROCm

Models using MXFP4 quantization with Mixture-of-Experts layers run significantly slower than expected on ROCm GPUs. The MoE computation falls back to dense per-expert execution instead of using fused kernels.

## How to reproduce

Run an MXFP4-quantized MoE model on a ROCm GPU and observe that MoE layer performance is far below what fused execution should achieve.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
