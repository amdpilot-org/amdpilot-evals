# Verification Record

- Status: `verified_with_caveat`
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

- the canonical bundle is structurally complete and reproducible
- there is **no reward-hacking evidence** from the run audit
- however, the current harness still relies too heavily on regex-style source
  checks, so a plausible-looking but not fully production-correct patch may
  satisfy it

## Next Action

Keep this bundle active, but strengthen the harness before treating it as a
fully hardened gold-standard success. The preferred direction is AST or runtime
validation rather than simple pattern matching.
