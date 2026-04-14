# SGLang Inference Produces Wrong Answers on FP8 Quantized Model

## Problem

When serving an FP8-quantized MoE model (Qwen3.5-397B-A17B-FP8) with sglang, the model produces incorrect or incoherent answers. The issue started after upgrading sglang to v0.5.9. Some responses are correct, but many contain wrong facts, garbled reasoning, or nonsensical output.

The problem is specific to the FP8 quantized computation path. Running the same model in BF16 produces correct results, suggesting a bug in sglang's FP8 MoE quantization pipeline — likely in how quantized weights, activations, or scaling factors are processed during the forward pass.

Symptoms observed:
- Simple factual questions get wrong answers (e.g., "What is 2+2?" returns wrong numbers)
- Longer generated text becomes incoherent or repetitive
- Results are non-deterministic — the same prompt may produce correct output sometimes and wrong output other times
- MoE expert outputs appear attenuated, as if the expert contributions are being suppressed

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
- Docker container with ROCm 7.2, PyTorch, AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands
- 4 GPUs needed for TP=4

## Where to Look

The bug is likely in sglang's FP8 MoE code path. Key areas to investigate:

1. **FP8 MoE quantization**: `sglang/srt/layers/moe/` — how FP8 weights and scales are prepared and passed to MoE kernels
2. **FP8 weight loading**: `sglang/srt/layers/quantization/fp8.py` — how FP8 model weights and per-block scales are loaded
3. **MoE kernel dispatch**: How sglang selects and invokes MoE kernels for FP8 models

Common FP8 MoE bugs to look for:
- Incorrect scaling factor application (wrong scale values, wrong dimensions)
- Buffer reuse or aliasing issues in the MoE output path
- Wrong dtype conversions losing precision
- Incorrect expert routing or weight selection

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness starts the server, sends diverse prompts, and checks output correctness and coherence. A score of 100.0 means all inference outputs are correct.

**Note**: Server startup takes 15-30 minutes due to model loading. The harness handles this automatically.
