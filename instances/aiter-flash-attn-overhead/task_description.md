# Performance: Reduce flash_attn_func Launch Overhead

## Problem

`aiter.flash_attn_func` has ~230us launch overhead per call due to a JIT module lookup
that always fails with an exception before falling through to the correct path.

## Root Cause

In `aiter/jit/core.py` (around line 818-823), the JIT dispatch always tries:

```python
if module is None:
    try:
        module = get_module(md_name)  # tries "module_mha_fwd" -> ALWAYS FAILS
    except Exception:
        md = custom_build_args.get("md_name", md_name)
        module = get_module(md)  # tries "mha_fwd_bf16_..." -> succeeds
```

1. `get_module("module_mha_fwd")` fails every call with `ModuleNotFoundError`
2. The exception handler then calls `get_module` with the correct variant name
3. This exception-throw-catch cycle adds ~150us overhead per call
4. The same pattern affects both forward and backward passes

## Impact

- Launch overhead: ~230us -> should be ~83us (2.8x improvement)
- For small attention computations (<100us kernel time), the overhead dominates
- Affects ALL users of `aiter.flash_attn_func`

## Working Directory

- AITER is at `/sgl-workspace/aiter` (symlinked to `/workspace/aiter`)
- The main file to fix is `aiter/jit/core.py`
- The fix should eliminate the failing `get_module` call path

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The test measures launch overhead before and after the fix. Target: <100us average launch overhead.
