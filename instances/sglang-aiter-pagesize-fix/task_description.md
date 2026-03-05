# Fix AITER Page-Size, DeepSeek MLA Tuple Inputs, and HiCache Backend Override

## Problem

Three cross-component bugs:

1. **aiter_backend.py**: MLA metadata hardcodes `page_size=1`, breaking `--page-size 64` configs.
2. **deepseek_v2.py**: `hidden_states` arrives as a tuple in quantization paths, crashing shape allocations.
3. **server_args.py**: HiCache workaround overrides user-selected `--attention-backend aiter` when it should only activate for FA3.

## Affected Files

- `python/sglang/srt/layers/attention/aiter_backend.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/server_args.py`

## Environment

- Use `/opt/venv/bin/python3` for all commands

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
