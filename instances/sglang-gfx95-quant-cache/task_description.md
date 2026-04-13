# Redundant gfx95 quant format detection in DeepseekV2DecoderLayer forward()

## Problem

In the DeepseekV2 model implementation, the `DeepseekV2DecoderLayer.forward()` method performs gfx95 quantization format detection on **every forward call**. This involves:

1. Checking whether the platform is gfx95
2. Inspecting weight tensors via `getattr()` chains to determine if they are `mxfp4` (`torch.uint8`) or `fp8` (`torch.float8_e4m3fn`)
3. Passing the result to `self.layer_communicator.prepare_attn()`

This detection logic produces the **same result every time** since the weight dtypes don't change after model loading. On gfx950 GPUs running DeepSeek-V2/V3 models with many layers (60+), the repeated per-call detection adds measurable overhead.

## Task

Optimize `DeepseekV2DecoderLayer` so the gfx95 quant format detection is performed at most once, not on every forward call. The result should be cached and reused.

Requirements:
- On non-gfx95 platforms, there should be zero runtime overhead
- On gfx95, the format should be detected once (it can be lazy — after weights are loaded)
- The `quant_format` value passed to `prepare_attn()` must be identical to the current logic

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
