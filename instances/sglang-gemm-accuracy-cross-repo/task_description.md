# SGLang Inference Produces Wrong Answers on FP8 Quantized Model

## Problem

When serving an FP8-quantized MoE model (e.g., Qwen3.5-397B-A17B-FP8) with sglang, the model produces incorrect or incoherent answers under sustained load. The issue is intermittent — some responses are correct, but many contain wrong facts, garbled reasoning, or nonsensical output.

The problem appears to be related to the FP8 quantized computation path. Running the same model in BF16 (without FP8 quantization) produces correct results, suggesting the issue is in how quantized weights or activations are processed during inference.

Symptoms observed:
- Simple factual questions get wrong answers (e.g., "What is 2+2?" returns wrong numbers)
- Longer generated text becomes incoherent or repetitive
- Results are non-deterministic — the same prompt may produce correct output sometimes and wrong output other times
- The issue is more pronounced under memory pressure (multiple requests, larger batches)

## Reproduction

Start the sglang server with an FP8 model and send inference requests:

```bash
# Start server (TP=4 for the 397B model)
/opt/venv/bin/python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/* \
    --tp 4 --attention-backend aiter --mem-fraction-static 0.80 --port 30000

# Send a request
curl http://localhost:30000/v1/chat/completions \
    -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"temperature":0}'
```

The response may contain wrong numerical answers, garbled text, or repeated tokens.

## Environment

- sglang serving framework at `/sgl-workspace/sglang`
- GPU compute libraries at `/sgl-workspace/aiter`
- Docker container with ROCm 7.2, PyTorch, AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands
- 4 GPUs needed for TP=4

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness starts the server, sends diverse prompts, and checks output correctness and coherence. A score of 100.0 means all inference outputs are correct.

**Note**: Server startup takes 15-30 minutes due to model loading. The harness handles this automatically.
