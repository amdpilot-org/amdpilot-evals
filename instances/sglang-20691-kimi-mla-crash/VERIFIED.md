# Verification Record

- Status: `preflight_blocked`
- Date: `2026-04-03`
- Verified by: `amdpilot-team`
- System version: `v0.4.0`
- Preflight score: `100.0`

## Result

Multiple reformulation attempts were blocked before agent trials because the
untouched baseline still returned `100/100`.

## Interpretation

Current evidence points to a reproduction gap rather than a clean SGLang-only
fix target:

- the tiny 2-layer Kimi test model does not reliably exercise the real fused MLA
  rope crash path
- the selected base image ships an older `aiter` path, so the intended backend
  route may silently fall back
- `issue open` was not enough evidence that the bug still existed on `main`

## Next Action

Treat this instance as formulation-blocked until the bug window, dependency
versions, and model path are all verified manually.
