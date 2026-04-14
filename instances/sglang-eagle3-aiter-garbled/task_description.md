# Fix EAGLE3 speculative decoding on aiter backend for non-MLA models

## Issue

EAGLE3 speculative decoding crashes or produces garbled output when using the
aiter attention backend on non-MLA models (e.g. Llama). The feature works
correctly on MLA models (e.g. DeepSeek) but breaks on standard attention
architectures.

### Error 1 -- AttributeError
```
AttributeError: 'AiterAttnBackend' object has no attribute 'max_split_per_batch'
```
This error occurs because code paths intended only for MLA models are being
executed unconditionally, accessing MLA-specific attributes that do not exist
on non-MLA model configurations.

### Error 2 -- Garbled output
Even when the AttributeError is avoided, EAGLE3 speculative decoding produces
garbled or incorrect output on non-MLA models. The CUDA graph capture and
replay paths use stale or incorrect tensor indices, leading to nonsensical
generation results.

## Environment

- SGLang source is available at `/sgl-workspace/sglang`
- Workspace symlink: `/workspace/sglang`
- Use `/opt/venv/bin/python3`
- Do **not** blank or clear `PYTHONPATH`
- Kill any leftover server processes with:

```bash
pgrep -f "python3 -m sglang" | xargs -r kill -9 2>/dev/null; sleep 2
```

## Verification

Run exactly:

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness reports a `SCORE` from `0` to `100`.

## Rules

- Edit source files only under `/sgl-workspace/sglang/`
- Do **not** modify `test_harness.py`
- Keep the patch minimal and localized
- Do **not** use broad `pkill` / `pgrep` patterns
