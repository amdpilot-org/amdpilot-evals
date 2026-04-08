# Fix AITER corruption in Qwen3.5-35B-A3B

## Issue

Historical vLLM issue `#35925`.

When the AITER stack is enabled for both attention and MoE, Qwen3.5-35B-A3B can
produce obviously corrupted outputs (for example long runs of repeated
punctuation like `!!!!!`), while the same prompts stay clean when AITER is
disabled.

## Environment

- vLLM source checkout: `/workspace/vllm`
- Python: `/usr/bin/python3`
- GPUs: 4x MI355X available in the container, but the harness reproduces with `TP=1`
- The preinstalled wheel has been removed; edits under `/workspace/vllm` are the runtime source of truth
- The image already ships with source-matched vLLM native extensions (`_C.abi3.so`, `_rocm_C.abi3.so`, `_moe_C.abi3.so`, `cumem_allocator.abi3.so`). If those imports fail, treat that as environment drift rather than the intended bug surface.

## Reproduction

The harness loads `Qwen/Qwen3.5-35B-A3B` with AITER-enabled environment flags and
checks both text-only and multimodal generations for corruption patterns.

## Verification

Run exactly:

```bash
export PYTHONPATH=/workspace/vllm
pgrep -f "python3 -m (sglang|vllm)" | xargs -r kill -9 2>/dev/null; sleep 2
/usr/bin/python3 /workspace/test_harness.py
```

The harness reports a `SCORE` from `0` to `100` based on whether corruption still
appears under the AITER-enabled path while the disabled baseline stays clean.

## Rules

- Edit files only under `/workspace/vllm`
- Do **not** modify `test_harness.py`
- Do **not** blank or clear `PYTHONPATH`
- Use the safe kill pattern shown above; never use broad `pkill` / `kill $(pgrep ...)`
- Do **not** treat rebuilding missing vLLM native extensions as the fix; this case is meant to test the AITER corruption path itself, not image bootstrap
- Keep the fix minimal and focused on the corruption path
