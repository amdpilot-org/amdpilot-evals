# Verification Record

- Status: `verified_success`
- Date: `2026-04-03`
- Verified by: `amdpilot-team`
- System version: `v0.4.0`
- Preflight score: `85.7`
- Verified score: `100.0`
- Trials: `1`

## Result

`sglang-19935-fp8-kscale-fallback` reached `100 verified` in a one-pass agent
run after the canonical bundle was reconstructed from the legacy
`sglang-sft` materials.

## Interpretation

- the canonical bundle is now structurally complete and reproducible
- the bug reproduces cleanly under the v0.4.0 unit-test-style flow
- the fix target is localized to the `mla_decode_fwd` k-scale fallback path

## Next Action

Keep this bundle in the active SGLang bugfix set and use it as the canonical
replacement for the older fragmented `sglang-19935-*` historical runs.
