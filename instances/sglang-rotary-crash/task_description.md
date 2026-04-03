# RotaryEmbedding fails on ROCm with CUDA_HOME error

ROCm CI fails during runtime initialization. On HIP, `RotaryEmbedding` dispatches through `MultiPlatformOp` and hits a fallback path that JIT-compiles with `cuda_files`. In ROCm-only environments, this triggers CUDA toolchain discovery and fails with:

```
RuntimeError: Could not find CUDA installation. Please set CUDA_HOME environment variable.
```

This breaks ALL models on ROCm since every model uses rotary embeddings.

Failing CI links:
- 2-GPU accuracy
- 1-GPU unit tests
- VLM accuracy and performance

## Affected File

- `python/sglang/srt/layers/rotary_embedding.py`

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Rules

- Edit files only under `/workspace/sglang`
- Use `/opt/venv/bin/python3` for all commands
- Do not modify `/workspace/test_harness.py`
- If you start auxiliary processes, never use broad kill patterns such as
  `pkill -f python` or `pkill -f sglang`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
