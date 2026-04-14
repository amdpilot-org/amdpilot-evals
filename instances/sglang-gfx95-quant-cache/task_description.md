# Redundant per-forward quantization format detection

## Problem

In the DeepSeek-V2/V3 model implementation, one of the decoder layer classes performs quantization format detection on **every forward call**. This involves inspecting weight tensor dtypes to determine the quantization scheme, then passing the result to the attention preparation path.

This detection logic produces the **same result every time** since the weight dtypes don't change after model loading. On gfx950 GPUs running models with many decoder layers (60+), the repeated per-call detection adds measurable overhead.

## Task

Optimize the decoder layer so that quantization format detection is performed at most once, not on every forward call. The result should be cached and reused.

Requirements:
- On non-gfx95 platforms, there should be zero runtime overhead
- On gfx95, the format should be detected once (it can be lazy — after weights are loaded)
- The quant format value passed to attention preparation must be identical to the current logic

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
