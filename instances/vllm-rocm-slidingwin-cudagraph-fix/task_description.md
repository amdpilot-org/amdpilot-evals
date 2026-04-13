# Bug: Sliding window models crash during CUDA graph capture on AITER FlashAttention backend

## Symptom

When running models with sliding window attention (e.g., Mistral) on the AITER FlashAttention backend on ROCm, CUDA graph capture crashes. The decode dispatch condition includes a `sliding_window` check that forces these models onto the `unified_attention` Triton path, which is incompatible with CUDA graph capture.

The decode dispatch logic incorrectly routes sliding window models to the `unified_attention` Triton path even during normal single-token decode, where they should use the standard paged attention path that supports CUDA graphs.

## How to Reproduce

1. Load a sliding window model (e.g., Mistral) on a ROCm GPU using the AITER FlashAttention backend.
2. Enable CUDA graph capture (default in vLLM v1).
3. Run decode inference — the sliding window condition forces the unified_attention path, which crashes during graph capture.

## Environment

- vLLM at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
