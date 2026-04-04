# Bug: AITER paged attention crashes for models with head_size < 64

## Symptom

When running models with attention head size smaller than 64 (e.g., head_size=32) on the AITER FlashAttention backend on ROCm, the `paged_attention_v1` ll4mi kernel crashes with an obscure kernel error. The ll4mi kernel requires `HEAD_SIZE >= 16 * NWARPS = 64`, but there is no fallback path for smaller head sizes.

## Affected File

- `vllm/v1/attention/backends/rocm_aiter_fa.py`

## How to Reproduce

1. Load a model with attention head size < 64 on a ROCm GPU using the AITER FlashAttention backend.
2. Run decode inference — the ll4mi paged attention kernel will crash because the head size doesn't meet the minimum requirement.
3. There is no automatic fallback to a compatible kernel path.

## Environment

- vLLM at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
