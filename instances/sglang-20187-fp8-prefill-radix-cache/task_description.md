# Enable FP8 prefill attention for the radix-cache path

## Issue

Historical SGLang issue `#20187`.

The aiter attention backend already supports FP8 prefill in the non-radix path,
but the radix-cache branch of `forward_extend()` falls back to BF16 and misses
the intended FP8 prefill speedup for DeepSeek-style models.

## Affected File

- `python/sglang/srt/layers/attention/aiter_backend.py`

## Symptom

The non-radix path of `forward_extend()` already handles FP8 prefill, but
when the radix cache is active, the code falls back to BF16 for the
attention computation. This means the FP8 prefill speedup is lost for
DeepSeek-style models in the radix-cache path.

The fix should be scoped to the prefill / radix-cache path in
`forward_extend()` only.

## Environment

- SGLang source is available at `/sgl-workspace/sglang`
- Workspace symlink: `/workspace/sglang`
- Use `/opt/venv/bin/python3`
- Do **not** blank or clear `PYTHONPATH`
- Safe cleanup pattern:

```bash
pgrep -f "python3 -m sglang" | xargs -r kill -9 2>/dev/null; sleep 2
```

## Verification

Run exactly:

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness checks whether FP8 prefill attention is integrated into the
radix-cache path with the expected fused GEMM + split/cat + FP8 handling.

## Rules

- Edit source files only under `/sgl-workspace/sglang/`
- Do **not** modify `test_harness.py`
- Keep the patch narrowly focused on the radix-cache FP8 prefill path
- Do **not** use broad `pkill` / `pgrep` patterns
