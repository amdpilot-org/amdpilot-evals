# vLLM #33123: Prefix caching output differs between cache miss and cache hit on ROCm

GitHub issue: https://github.com/vllm-project/vllm/issues/33123

## Why this eval bundle exists

This issue is still open and is known to be prone to weak / flaky reproduction.
The eval bundle therefore uses:

- a pinned pre-fix checkout
- source-over-wheel runtime forcing
- three prompt lengths (different token/block remainders)
- repeated requests + logprob comparison
- a prefix-caching-disabled negative control

The goal is to avoid false `score=100` results that only happen because the
bug was not exercised strongly enough.

## Reproduction target

The bug is: with prefix caching enabled, request 1 (cache miss) produces a
different result than later identical requests (cache hit).

The issue is reported for ROCm MI355X / bf16 and affects both Triton and ROCm
AITER attention paths, suggesting a shared cache/prefix path issue.

## Environment

- Use `/usr/bin/python3`
- Runtime must import from `/workspace/vllm`, not the pre-installed wheel
- Model: `Qwen/Qwen3-0.6B`
- Single GPU ROCm MI355X

## Verification

Run:

```bash
unset PYTHONPATH
/usr/bin/python3 /workspace/test_harness.py
```

The harness:
1. starts a vLLM server with prefix caching enabled
2. tests 3 prompts of different lengths
3. sends 6 requests per prompt
4. checks both output text and per-token logprob stability
5. reruns the same prompts with prefix caching disabled as a negative control

Success means:
- all requests complete
- all outputs are identical
- repeated cache-hit requests remain stable

## Rules

- Edit `/workspace/vllm`
- Do not modify the harness
- Do not lower the request count
- Before any command, `unset PYTHONPATH`
