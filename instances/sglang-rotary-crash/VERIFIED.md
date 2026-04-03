# Verification Record

- Status: `verified_success`
- Date: `2026-04-03`
- Verified by: `amdpilot-team`
- System version: `v0.4.0`
- Preflight score: `66.7`
- Verified score: `100.0`
- Trials: `1`

## Result

`sglang-rotary-crash` reproduced on preflight and then reached `100 verified`
in a one-pass agent run on v0.4.0.

## Interpretation

- this issue does not depend on `aiter` fused MLA support
- the Feb 26 `sgl-dev` base image is sufficient for reproduction
- unit-test-style harnesses are working well for this class of SGLang bugfix

## Next Action

Keep this bundle in the active, reusable bugfix set and use it as a reference
pattern for future lightweight SGLang instances.
