# Unified Attention Crashes on Non-Power-of-2 Block Size

## Context

The unified attention kernel in `aiter/ops/triton/attention/unified_attention.py` crashes when used with models that have a non-power-of-2 block size, such as Qwen3-Next which uses `block_size=48`.

The functions `select_2d_config` and `select_3d_config` set `TILE_SIZE` directly from the `block_size` parameter. Triton requires `TILE_SIZE` to be a power of 2 for correct kernel compilation and execution. When `block_size` is not a power of 2 (e.g. 48), the resulting `TILE_SIZE` value causes the Triton compiler to crash during kernel compilation.

## Affected Files

- `aiter/ops/triton/attention/unified_attention.py`

## Environment

- AITER at `/sgl-workspace/aiter`
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
