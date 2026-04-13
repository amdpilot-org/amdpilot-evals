# Kimi-K2.5 W4A16 MoE CK Kernel Support on ROCm

## Context

Kimi-K2.5 is a 1T-parameter MoE model (DeepSeek V3 architecture) with 384
experts (8 active per token), using INT4 weight-only quantization (W4A16)
via the `compressed-tensors` format. It is one of the largest publicly
available MoE models and a key target for AMD MI355X inference.

## Current State

The model loads and runs inference on ROCm, but the MoE layers fall back to
a **Triton-based kernel path** because the CK (Composable Kernel) MoE
backend in aiter does not support W4A16 quantized weights
(int4-packed-as-uint32). When aiter's `fused_moe` is called with uint32
weight tensors, it fails with:

```
RuntimeError: get_cfg Unsupported input_type:BFloat16, weight_type:Int,
out_type:BFloat16, quant_type:0, do_weight:0
```

The CK MoE 2-stage pipeline in aiter currently supports bf16, fp16, fp8,
fp4, and int8 weight formats — but NOT int4/uint32 (W4A16). The Triton
fallback works but does not leverage the optimized CK kernel pipeline.

## What Has Already Been Done (in the checkpoint)

Previous trials solved several integration steps. These are already in
place in the container — do NOT redo them:

1. **CK codegen** (aiter): W4A16 kernel templates and heuristic dispatch
   entries have been added. The compiled `.so` modules contain A16W4
   kernel symbols.

2. **SGLang routing class**: A W4A16-specific aiter MoE scheme class
   has been created with an `apply_weights()` method.

3. **Weight loading**: The W4A16 aiter MoE scheme has been added to the
   transpose allowlists in the MoE layer configuration.

4. **Import/export**: The W4A16 aiter MoE scheme is properly exported
   from the schemes package.

5. **Dtype detection**: int32/uint32 handling has been added to the
   aiter fused MoE dispatch logic.

**NOT done yet**: The routing logic on HIP still returns the Triton MoE
scheme instead of the aiter one.

## What Failed and Why

- **bf16 dequant approach (OOM)**: Dequantizing all 384 expert weights
  from int4 to bf16 before calling `aiter.fused_moe()` caused HIP OOM.
  384 experts × 4x memory = ~240GB additional per GPU. This model uses
  ~273GB VRAM on 8×288GB GPUs with no room for dequant copies.
  **Do NOT attempt full-model dequant again.**

- **QuantType.per_1x32**: Triggers FP4 activation quantization in the
  fused_moe pipeline, wrong for W4A16 (bf16 activations + int4 weights).

- **Generic CK path deadlock**: Routing `torch.int32` weights through
  the generic CK 2-stage MoE pipeline causes a **DEADLOCK** during
  CUDA graph capture (0% GPU utilization, workers hang indefinitely).
  **Do NOT route W4A16 through the generic CK condition.**

- **ASM path (wrong target)**: The ASM MoE path needs C++ JIT
  recompilation and has an Int8 placeholder config. This is NOT the
  correct target for W4A16.
  **Do NOT attempt C++ changes to the ASM MoE kernel sources.**

## Task: Add Native W4A16 Support to CK MoE Dispatch

The Triton MoE fallback works with packed int4 weights directly — it
passes uint32 tensors to the kernel, which handles dequant internally.
The CK path needs the same: native `weight_type:Int` support without
creating temporary bf16 copies.

### Important Notes

- Kimi-K2.5 uses `torch.int32` for packed int4 weights, NOT
  `torch.uint32`. Many code paths only check uint32 — you must handle
  both.
- One of the CK MoE `.so` modules already has compiled A16W4 kernels.
  No codegen or C++ compilation is needed.
- Do NOT add `torch.int32` to the generic CK 2-stage condition — that
  path deadlocks during CUDA graph capture.

## Environment

- SGLang editable install at `/sgl-workspace/sglang/`
- AITER library at `/sgl-workspace/aiter/` (CK MoE kernels live here)
- Model weights cached at `/root/.cache/huggingface/Kimi-K2.5`
- Docker container with ROCm, PyTorch, 8x AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands
- Working directory must be `/root` (not `/sgl-workspace`) to avoid aiter
  namespace package import conflicts

## Benchmark

The benchmark script at `/workspace/bench_kimi_w4a16.sh` runs
`sglang.bench_one_batch` with fixed parameters. Do NOT modify the benchmark
parameters (model, tp, batch size, input/output lengths).

A lightweight smoke test at `/workspace/bench_kimi_w4a16_smoke.sh` uses
`--output-len 2` instead of 2048 — use it for fast iteration (~3-5 min vs
15-30 min) to validate that the model loads and the MoE path executes
correctly before running the full benchmark.

You may write `/workspace/bench_config.env` to configure environment
variables (e.g., backend selection flags). This file is sourced by both
benchmark scripts before each run.

```bash
# Fast smoke test (validates model loads + single decode step):
bash /workspace/bench_kimi_w4a16_smoke.sh

# Full benchmark (measures decode latency):
bash /workspace/bench_kimi_w4a16.sh
```

The benchmark outputs `Decode median (ms): <value>` — lower is better.

## Rules

- Do not use `pkill -f` to kill processes
- CUDA graphs must remain enabled (do not disable them)
- Do NOT attempt to dequantize all expert weights to bf16 (causes OOM)
- Do NOT revert routing to Triton once CK path is activated
- Run `bash /workspace/bench_kimi_w4a16.sh` as the last command to verify
  your changes
