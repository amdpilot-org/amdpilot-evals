# MoE Expert Routing Crash at Long Sequence Lengths

## Problem

When serving large MoE models with a non-standard number of experts (e.g., 384) on AMD GPUs, the system crashes at long input lengths but works fine at shorter ones.

For example, a model with 384 experts and grouped top-k routing runs correctly for inputs up to ~32K tokens, but crashes with a `RuntimeError` when input length exceeds ~55K tokens. The crash occurs in the expert routing / top-k selection step of the MoE layer.

## Task

Fix the expert routing in the AITER library so that models with arbitrary expert counts can serve long sequences without crashing. The fix should not degrade performance for models that already work correctly.

## Environment

- AITER at `/sgl-workspace/aiter` (also symlinked at `/workspace/aiter`)
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
