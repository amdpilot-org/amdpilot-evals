# MoE expert routing produces corrupted outputs on ROCm

## Symptom

When running Mixture-of-Experts models on ROCm GPUs, the expert routing layer silently corrupts the token-to-expert assignment. Some tokens are routed to incorrect experts, and valid token-expert pairs are dropped. This leads to nonsensical model output or GPU memory access faults during inference.

The corruption is deterministic on ROCm but does not occur on CUDA GPUs with the same model and inputs.

## How to reproduce

Run an MoE model (e.g., any model using fused MoE with expert routing) on a ROCm GPU and compare the output quality against the same model on CUDA. The ROCm output will be garbage or the process will crash with an illegal memory access.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
