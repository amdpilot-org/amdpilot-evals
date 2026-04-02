# vLLM #35925: Qwen3.5-35B-A3B corrupted responses when AITER is enabled

GitHub issue: https://github.com/vllm-project/vllm/issues/35925

## Why this eval bundle exists

The issue was later fixed upstream by a stride-related ROCm patch. To avoid
false positives, this eval bundle pins the container checkout to the commit
immediately **before** that fix and installs vLLM from the checked-out source.

This bundle should answer one question only:

> On a known pre-fix ROCm vLLM commit, can the candidate patch make AITER
> outputs stop producing corrupted text?

## Reproduction target

The original issue reports corruption when AITER is enabled for Qwen3.5-35B-A3B.
The corruption shows up as repetitive punctuation or clearly garbled outputs.
The report started with multimodal evaluation and later confirmed text-only
repro as well.

## Environment

- Use `/usr/bin/python3`
- Runtime must import from `/workspace/vllm`, not the pre-installed wheel
- Model: `Qwen/Qwen3.5-35B-A3B`
- GPUs: 4x AMD MI355X (TP=4)

## Verification

Run:

```bash
unset PYTHONPATH
/usr/bin/python3 /workspace/test_harness.py
```

The harness runs:
1. a text-only baseline with AITER disabled
2. a text-only run with AITER enabled
3. a multimodal run with AITER enabled

Success means:
- all baseline outputs are clean
- no AITER-enabled output is classified as corrupted
- all requests complete successfully

## Rules

- Edit `/workspace/vllm`
- Do not modify the harness
- Do not change prompt counts, model choice, or corruption criteria
- Before any command, `unset PYTHONPATH`
