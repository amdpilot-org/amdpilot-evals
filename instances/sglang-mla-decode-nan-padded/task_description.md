# Intermittent NaN in MLA Decode Output with Speculative Decoding

## Problem

When serving a large MoE model with MLA (Multi-head Latent Attention) on AMD GPUs using the aiter attention backend with speculative decoding enabled, the model intermittently produces NaN values in its output. This causes corrupted or incomplete responses.

The NaN propagation is non-deterministic — it depends on what happens to be in uninitialized GPU memory at the time of the decode step. Some requests may succeed while others produce NaN-corrupted output. The issue only manifests during the decode phase (not prefill) and is related to how the attention backend handles padded sequence batches.

## Reproduction

```bash
# Start server with aiter backend and speculative decoding
SGLANG_ENABLE_SPEC_V2=1 /opt/venv/bin/python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b \
  --tp 2 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --attention-backend aiter \
  --mem-fraction-static 0.8 \
  --port 30000

# Send multiple requests — some will produce NaN-corrupted output
for i in $(seq 1 10); do
  curl -s http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"default\",\"messages\":[{\"role\":\"user\",\"content\":\"What is $i + $i?\"}],\"temperature\":0,\"max_tokens\":64}"
done
```

## Environment

- SGLang at `/sgl-workspace/sglang`
- GPU compute libraries at `/sgl-workspace/aiter`
- Docker container with ROCm 7.2, PyTorch, AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands
- Model weights are pre-loaded

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness starts the server with speculative decoding, sends inference requests, and checks whether outputs contain NaN or produce coherent text. A score of 100.0 means all responses are valid.

**Important**: Server startup includes model loading which takes significant time. Verify your changes are correct before running the full test.
