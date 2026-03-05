# Fix EAGLE3 Speculative Decoding with AITER Attention Backend

## Problem

EAGLE3 speculative decoding crashes when using the aiter attention backend. The `target_verify` CUDA graph has stale `qo_indptr`/`kv_indptr`/`kv_indices`, and there are missing `use_mla` guards that cause `AttributeError` on non-MLA models. The capture/replay logic needs restructuring.

## Affected Files

- `python/sglang/srt/layers/attention/aiter_backend.py`

## Environment

- Use `/opt/venv/bin/python3` for all commands

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
