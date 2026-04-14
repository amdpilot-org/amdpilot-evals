# AMD attention performance regression with non-default context-length

## Problem

On AMD GPUs (ROCm/HIP), running models with `--context-length` set to less than 32768 causes a **~40% end-to-end performance drop** during decode.

The attention backend's decode path selects between different kernel implementations based on the maximum KV cache length. With certain context-length settings, the kernel dispatcher incorrectly selects a slower path even for long sequences, resulting in severe performance degradation.

For example, with `context-length=13824`, long sequences should use the high-performance path, but the system falls back to a slower kernel for the entire decode.

## Task

Find and fix the bug in the attention backend's decode path that causes incorrect kernel selection when the context-length is reduced from the default.

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
