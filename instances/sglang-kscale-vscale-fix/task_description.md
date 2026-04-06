# extend_attention_fwd() Crashes with Missing k_scale / v_scale Arguments

When running non-MLA speculative decoding (target_verify / draft_extend paths) on AMD ROCm GPUs with hybrid models like Qwen3-Coder-Next using MTP, the attention forward pass crashes with:

```
TypeError: extend_attention_fwd() missing 2 required positional arguments: 'k_scale' and 'v_scale'
```

The error occurs in the `forward_extend` method of the aiter attention backend:

- `python/sglang/srt/layers/attention/aiter_backend.py`

The function `extend_attention_fwd` (defined in `triton_ops/extend_attention.py`) expects `k_scale` and `v_scale` as required positional parameters, but the call site in `aiter_backend.py` does not pass them.

## Affected File

- `python/sglang/srt/layers/attention/aiter_backend.py`

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
