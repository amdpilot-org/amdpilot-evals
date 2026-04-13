# Qwen3.5-397B-A17B-FP8 Produces Garbage Output with EAGLE Speculative Decoding

## Problem

When serving Qwen3.5-397B-A17B-FP8 with EAGLE speculative decoding enabled, the model produces garbled/garbage output instead of coherent responses. The output contains nonsensical character sequences, repeated patterns, and corrupted text.

The issue is reproducible with concurrent requests and appears to be related to how compute operations are dispatched during speculative decode inference.

## Reproduction

Start the server:
```bash
SGLANG_ENABLE_SPEC_V2=1 /opt/venv/bin/python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-397B-A17B-FP8 \
  --tp 4 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-aiter-allreduce-fusion \
  --attention-backend triton \
  --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --port 30000
```

Send concurrent requests and observe garbage output in the responses.

## Environment

- SGLang source: `/sgl-workspace/sglang` (symlinked at `/workspace/sglang`)
- GPU compute library: `/sgl-workspace/aiter` (symlinked at `/workspace/aiter_repo`)
- Use `/opt/venv/bin/python3`
- Kill leftover servers: `pkill -f sglang.launch_server; sleep 2`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness starts the server, sends 8 concurrent requests, and checks responses for garbage patterns. Score of 100.0 means all responses are coherent.

## Rules

- Edit source files under `/sgl-workspace/sglang/` or `/sgl-workspace/aiter/`
- Do **not** modify `test_harness.py`
- Keep the fix minimal and targeted
- If modifying C/C++ source, clear relevant build caches and recompile
