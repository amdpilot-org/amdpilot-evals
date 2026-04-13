# AMD attention performance regression with non-default context-length

## Problem

On AMD GPUs (ROCm/HIP), running models with `--context-length` set to less than 32768 causes a **~40% end-to-end performance drop** during decode.

The issue is in the aiter attention backend's decode path (the aiter attention backend). The `unified_attention` kernel selects between a 2D kernel (for short sequences, `max_seqlen_k <= 512`) and a 3D kernel based on the maximum KV cache length. With certain context-length settings, the kernel dispatcher incorrectly selects the slower 2D path even for long sequences, resulting in severe performance degradation.

For example, with `context-length=13824`, the system should use the 3D kernel for sequences above 512 tokens, but instead falls back to the slower 2D kernel for the entire decode.

## Task

Find and fix the bug in the aiter attention backend's decode path that causes incorrect kernel selection when the context-length is less than 32768.

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
