# Verification Record

- Status: `preflight_passed`
- Date: `2026-04-03`
- Verified by: `amdpilot-team`
- System version: `v0.4.0`
- Preflight score: `75.0`

## Result

The instance reproduced the bug strongly enough to pass preflight and enter an
agent trial on the v0.4.0 system.

## Interpretation

- this issue does not depend on `aiter` fused MLA support
- the Feb 26 `sgl-dev` base image is sufficient for reproduction
- this is a clean SGLang bugfix candidate under the new flow

## Next Action

Keep the bundle in the active bugfix set and update this file with the first
verified rerun result after the agent completes.
