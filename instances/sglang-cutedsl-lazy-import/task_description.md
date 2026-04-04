# kda_backend Crashes on ROCm with ModuleNotFoundError

Importing the KDA attention backend crashes on AMD ROCm GPUs with:

```
ModuleNotFoundError: No module named 'cuda'
```

The error occurs when importing `sglang.srt.layers.attention.linear.kda_backend`. The module-level import chain pulls in CuteDSL, which depends on `cuda.bindings` -- a package that only exists on NVIDIA CUDA platforms. This makes the entire `kda_backend` module unimportable on ROCm, even if KDA is never actually used.

## Affected File

- `python/sglang/srt/layers/attention/linear/kda_backend.py`

## Reproduction

```bash
/opt/venv/bin/python3 -c "from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend"
```

This crashes immediately with `ModuleNotFoundError: No module named 'cuda'` on ROCm.

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
