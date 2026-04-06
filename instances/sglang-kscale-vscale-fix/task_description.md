# Attention Forward Pass Crashes During Speculative Decoding on ROCm

When running non-MLA speculative decoding (target_verify / draft_extend paths) on AMD ROCm GPUs with hybrid models like Qwen3-Coder-Next using MTP, the attention forward pass crashes with:

```
TypeError: extend_attention_fwd() missing 2 required positional arguments: 'k_scale' and 'v_scale'
```

The crash occurs during the extend attention path in the aiter attention backend. A function call is missing required positional arguments.

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
