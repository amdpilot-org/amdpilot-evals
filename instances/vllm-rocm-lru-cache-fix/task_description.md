# ROCm sparse MLA: paged MQA logits helper not cached

On ROCm, the FP8 paged MQA logits path for sparse MLA (`paged_mqa_logits` integration) reloads the underlying Triton / aiter module on every call instead of reusing a stable cached import. That defeats `lru_cache` and causes repeated import work, hurting performance for ROCm sparse MLA (e.g., DeepSeek v3.2 style flows).

## Affected area

- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
