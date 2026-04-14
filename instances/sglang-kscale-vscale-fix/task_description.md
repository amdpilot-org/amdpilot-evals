# extend_attention_fwd() Crashes with Missing k_scale / v_scale Arguments

When running non-MLA speculative decoding (target_verify / draft_extend paths) on AMD ROCm GPUs with hybrid models like Qwen3-Coder-Next using MTP, the attention forward pass crashes with:

```
TypeError: extend_attention_fwd() missing 2 required positional arguments: 'k_scale' and 'v_scale'
```

The error occurs during the attention forward pass in the aiter backend on ROCm.

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
