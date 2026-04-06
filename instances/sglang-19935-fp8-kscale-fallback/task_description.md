# Fix FP8 MLA decode `k_scale` fallback in `aiter_backend.py`

## Issue

Historical SGLang issue `#19935`.

When `layer.k_scale` is `None` in the aiter MLA decode path, all four
`mla_decode_fwd` call sites in `python/sglang/srt/layers/attention/aiter_backend.py`
pass that `None` through directly instead of falling back to `self.k_scale`.
This causes the FP8 MLA decode path to fail.

## Affected File

- `python/sglang/srt/layers/attention/aiter_backend.py`

## Symptom

When running a model that does NOT use FP8, the `layer.k_scale` attribute is
`None`. The four `mla_decode_fwd` call sites in `aiter_backend.py` pass
this `None` through without any guard, causing a downstream assertion or
TypeError at runtime.

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

The harness statically and semantically checks the four `mla_decode_fwd`
call sites and reports a `SCORE` from `0` to `100`.

## Rules

- Edit source files only under `/sgl-workspace/sglang/`
- Do **not** modify `test_harness.py`
- Keep the patch minimal and localized
- Do **not** use broad `pkill` / `pgrep` patterns
