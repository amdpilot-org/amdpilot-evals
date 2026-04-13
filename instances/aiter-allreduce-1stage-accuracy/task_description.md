# Fused Allreduce+RMSNorm Produces Incorrect Numerical Results

## Problem

When running a large MoE model with tensor parallelism on AMD GPUs, the fused allreduce+RMSNorm kernel produces slightly different numerical results compared to the unfused (separate allreduce then RMSNorm) path. The per-element error is small (1 ULP in bf16), but it compounds across transformer layers. For a 60-layer model at TP=4, this causes measurable accuracy regression (e.g., ~2.6 percentage points drop on GSM8K benchmark).

The issue only affects the 1-stage fused kernel path, which is selected when the total reduction data fits within a small buffer (approximately ≤128KB). This typically occurs during decode with small batch sizes (batch=1). The 2-stage kernel path is unaffected.

## Reproduction

The bug can be reproduced by comparing the output of the fused allreduce+RMSNorm operation against the unfused equivalent:

```python
# Pseudocode:
# 1. Initialize a 2+ GPU process group
# 2. Create bf16 input tensors (hidden_states, residual, rmsnorm weight)
# 3. Run fused allreduce+rmsnorm (aiter custom_all_reduce)
# 4. Run unfused: nccl allreduce → add residual → rmsnorm
# 5. Compare outputs — they should match, but ~25% of elements diverge by 1+ ULP
```

The divergence stems from the fused kernel's numerical sequence differing from the unfused path in how it handles intermediate precision conversions.

## Environment

- GPU compute libraries at `/sgl-workspace/aiter`
- Docker container with ROCm 7.2, PyTorch, AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands
- Requires 2+ GPUs for allreduce operations

## Verification

Run the test harness after applying your fix:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

The harness compares fused vs unfused allreduce+rmsnorm outputs across multiple random inputs and checks for numerical equivalence. A score of 100.0 means the fused and unfused paths produce bit-identical results.

**Important**: The fix is in C/C++ source code. After modifying source files, you must recompile for changes to take effect. JIT caches under `aiter/jit/` should be cleared before recompilation.
