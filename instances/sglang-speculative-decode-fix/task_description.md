# Qwen3.5-397B-A17B-FP8 Generates Garbage Output with Speculative Decoding

## Problem

When serving the Qwen3.5-397B-A17B-FP8 model with EAGLE speculative decoding enabled on ROCm, the server fails to produce coherent output. The failure may manifest as a server deadlock (no response), corrupted/garbage output with repeated characters and random strings, or a GPU hardware fault.

The issue is triggered by the speculative decoding execution path on ROCm.

## Reproduction

Start the server with EAGLE speculative decoding:
```bash
SGLANG_ENABLE_SPEC_V2=1 /opt/venv/bin/python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b \
  --tp 2 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-aiter-allreduce-fusion \
  --attention-backend triton \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --reasoning-parser qwen3 \
  --port 30000
```

Send any chat completion request:
```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Explain the theory of relativity in simple terms."}],"temperature":0,"max_tokens":512}'
```

The response will contain garbage text instead of a coherent answer.

## Environment

- SGLang at `/sgl-workspace/sglang`
- GPU compute libraries at `/sgl-workspace/aiter`
- Docker container with ROCm 7.2, PyTorch, AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands
- Model weights are pre-loaded and available via the standard model path

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness starts the server with speculative decoding, sends inference requests, and checks whether the output is coherent. A score of 100.0 means all responses are correct.

**Important**: After modifying source code, you may need to clear JIT-compiled caches and restart the server for changes to take effect. Server startup includes model loading which takes significant time, so verify your changes are correct before running the full test. For C/C++ changes, JIT caches under `aiter/jit/` must be cleared before restart.
