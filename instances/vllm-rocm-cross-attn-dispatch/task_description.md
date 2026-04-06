# ROCm encoder-decoder: wrong beam search on ROCM_ATTN / ROCM_AITER_FA

Encoder-decoder models (for example Whisper, BART) produce incorrect beam search results on ROCm when attention is routed to the ROCM_ATTN or ROCM_AITER_FA backends. The failure shows up when more than one query position is active at once (`max_query_len > 1`), as in beam search, because cross-attention is handled with kernels that assume decoder self-attention semantics.

## Affected area

- `vllm/platforms/rocm.py`
- `vllm/v1/attention/backends/rocm_attn.py`
- `vllm/v1/attention/backends/rocm_aiter_fa.py`

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
