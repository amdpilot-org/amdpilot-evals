# /generate Endpoint Crashes on Non-Finite top_logprobs Values

## Context

The `/generate` endpoint fails with a JSON serialization error when the model produces non-finite float values (NaN, -Infinity, Infinity) in `top_logprobs`. This occurs because ORJSON's default serialization mode rejects non-finite floats, but the `/generate` endpoint does not use the appropriate ORJSON options to handle them.

The OpenAI-compatible endpoints (`/v1/chat/completions`, etc.) are unaffected because they use a different serialization path. Only the native `/generate` endpoint is broken.

## Environment

- SGLang at `/sgl-workspace/sglang`
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
