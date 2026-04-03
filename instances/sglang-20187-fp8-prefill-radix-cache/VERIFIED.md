# Verification Record

- Status: `verified_success`
- Date: `2026-04-03`
- Verified by: `amdpilot-team`
- System version: `v0.4.0`
- Preflight score: `82.4`
- Verified score: `100.0`
- Trials: `1`

## Result

`sglang-20187-fp8-prefill-radix-cache` reached `100 verified` in a one-pass
agent run after the canonical bundle was reconstructed from the legacy
`sglang-sft` materials.

## Interpretation

- the canonical bundle is now structurally complete and reproducible
- the bug reproduces cleanly under the v0.4.0 unit-test-style flow
- the fix target is localized to the FP8 prefill radix-cache path

## Next Action

Keep this bundle in the active SGLang bugfix set and use it as the canonical
replacement for the older fragmented `sglang-20187-*` historical runs.
