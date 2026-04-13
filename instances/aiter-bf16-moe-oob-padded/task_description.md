# BF16 CK 2-Stage MoE Kernel OOB on Padded Rows

## Context

The CK (Composable Kernel) 2-stage MoE GEMM kernel is used for BF16
mixture-of-experts inference on AMD MI355X GPUs. It processes tokens
dispatched to each expert via `moe_sorting`, which produces sorted
token IDs, expert IDs, and per-token weights.

When the number of tokens assigned to an expert does not fill the last
block, `moe_sorting` pads the remaining slots with sentinel rows. These
sentinel entries have token IDs that point beyond the valid token range
and weights that are set to zero.

## Problem

Running `sglang.bench_one_batch` with a BF16 MoE model (e.g.,
`Qwen/Qwen3-Next-80B-A3B-Instruct`) on this platform crashes with a
GPU memory access fault:

```
Memory access fault by GPU node-3 on address 0x7d4607000000.
Reason: Write access to a read-only page.
```

The crash occurs during the MoE forward pass. The CK 2-stage kernel
processes sentinel/padded rows without checking whether the token
offset is within the valid range, leading to out-of-bounds memory
access when the sentinel token ID maps to an unmapped GPU memory region.

## Task

Fix the CK 2-stage MoE kernel so that it correctly handles
sentinel/padded rows without accessing out-of-bounds memory. The fix
must be in the kernel itself — consumer-side workarounds (clamping
sorted IDs before dispatch, switching to a different kernel backend)
are not acceptable.

## Environment

- AITER library: `/sgl-workspace/aiter/` (editable install)
- Composable Kernel (CK) headers: `/sgl-workspace/aiter/3rdparty/composable_kernel/`
- SGLang: `/sgl-workspace/sglang/` (editable install)
- Model weights: `/root/.cache/huggingface/`
- Docker container with ROCm, PyTorch, 8x AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all Python commands

The CK MoE kernel is JIT-compiled on first use. After modifying kernel
source files, delete the JIT cache at `/sgl-workspace/aiter/aiter/jit/build/`
to force recompilation.

## Benchmark

```bash
bash /workspace/bench_bf16_moe.sh
```

The benchmark runs `sglang.bench_one_batch` with fixed parameters.
It outputs `SCORE: 100.0` on success or `SCORE: 0.0` on failure.

## Rules

- Do not use `pkill -f` to kill processes
- Do not modify the benchmark script or test harness
- The fix must be in the CK kernel source, not in consumer code
- Run `bash /workspace/bench_bf16_moe.sh` as the last command to verify
