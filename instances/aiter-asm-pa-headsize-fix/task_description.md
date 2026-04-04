# ASM Paged Attention Produces Incorrect Results for head_size != 128

## Context

The ASM paged attention kernel in `aiter/ops/triton/attention/attention.py` only supports `head_size=128`, but the dispatch logic (`_should_use_asm_kernel`) does not check the head size before routing to the ASM kernel path. When a model uses a different head size (e.g. `head_size=64`), the ASM kernel is incorrectly selected and produces wrong attention output values.

The attention results are silently incorrect rather than raising an error, making this a data correctness bug that can go undetected until the model produces garbled output.

## Affected Files

- `aiter/ops/triton/attention/attention.py`

## Environment

- AITER at `/sgl-workspace/aiter`
- Docker container with ROCm, PyTorch, AMD GPU
- Use `/opt/venv/bin/python3` for all commands

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
