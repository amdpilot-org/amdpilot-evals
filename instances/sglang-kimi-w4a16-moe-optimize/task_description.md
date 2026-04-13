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

The CK MoE 2-stage pipeline (`ck_moe_stage1` / `ck_moe_stage2` in aiter)
currently supports bf16, fp16, fp8, fp4, and int8 weight formats — but NOT
int4/uint32 (W4A16). The Triton fallback works but does not leverage the
optimized CK kernel pipeline.

## What Has Already Been Done (in the checkpoint)

Previous trials solved several integration steps. These are already in
place in the container — do NOT redo them:

1. **CK codegen** (aiter): W4A16 kernel templates added in
   `gen_instances.py` (+131 lines) and `gemm_moe_ck2stages_common.py`
   (+51 lines). Templates `A16W4_gemm1_heuristic_dispatch` and
   `A16W4_gemm2_heuristic_dispatch` exist.

2. **SGLang routing class**: `CompressedTensorsWNA16AiterMoE` class
   created in `compressed_tensors_wNa16_moe.py` (+79 lines) with
   `apply_weights()` method.

3. **Weight loading**: `CompressedTensorsWNA16AiterMoE` added to both
   transpose allowlists in `fused_moe_triton/layer.py` (lines ~703 and
   ~923).

4. **Import/export**: `CompressedTensorsWNA16AiterMoE` exported in
   `schemes/__init__.py`.

5. **Dtype detection**: int32/uint32 handling added in
   `aiter/fused_moe.py`.

**NOT done yet**: The routing logic in `compressed_tensors.py` still
returns the Triton MoE scheme on HIP instead of the aiter one.

## What Failed and Why

- **bf16 dequant approach (OOM)**: Dequantizing all 384 expert weights
  from int4 to bf16 before calling `aiter.fused_moe()` caused HIP OOM.
  384 experts × 4x memory = ~240GB additional per GPU. This model uses
  ~273GB VRAM on 8×288GB GPUs with no room for dequant copies.
  **Do NOT attempt full-model dequant again.**

- **QuantType.per_1x32**: Triggers FP4 activation quantization in the
  fused_moe pipeline, wrong for W4A16 (bf16 activations + int4 weights).

- **Generic CK path deadlock**: Adding `torch.int32` to the generic CK
  2-stage condition in `get_2stage_cfgs()` routes through
  `ck_moe_stage1` → `module_moe_ck2stages.so`. Despite having A16W4
  kernel symbols, this path **DEADLOCKS** during CUDA graph capture with
  `torch.int32` weights (0% GPU utilization, workers hang indefinitely).
  **Do NOT route W4A16 through the generic CK condition.**

- **ASM path (wrong target)**: The ASM path (`asm_stage1` →
  `module_moe_asm.so`) needs C++ JIT recompilation and has an Int8
  placeholder config. This is NOT the correct target for W4A16.
  **Do NOT attempt C++ changes to asm_moe_2stage.cu.**

## Task: Add Native W4A16 Support to CK MoE Dispatch

The Triton MoE fallback works with packed int4 weights directly — it
passes uint32 tensors to the kernel, which handles dequant internally.
The CK path needs the same: native `weight_type:Int` support without
creating temporary bf16 copies.

### Remaining Steps

There are THREE kernel paths in aiter's MoE dispatch. Only the **CK tile
path** works for W4A16:

| Path | Functions | .so file | W4A16 status |
|------|-----------|----------|--------------|
| CK tile | `cktile_moe_stage1` / `cktile_moe_stage2` | `module_moe_cktile2stages.so` (4MB) | **WORKS** — has compiled A16W4 kernels |
| Generic CK | `ck_moe_stage1` | `module_moe_ck2stages.so` (170MB) | **DEADLOCKS** with int32 weights |
| ASM | `asm_stage1` | `module_moe_asm.so` | Wrong config, needs recompilation |

The fix is **Python-only** — no C++ changes or recompilation needed.

1. **Route W4A16 through CK tile path** (`fused_moe.py`): Add dispatch
   logic in `get_2stage_cfgs()` so that int4-packed weight tensors
   (uint32/int32) are routed through the CK tile 2-stage functions
   instead of the generic CK path. The branch must go BEFORE the
   generic CK condition.

2. **Fix dimension calculation** (`fused_moe.py`): The CK tile stage
   functions need to handle int4-packed weight dimensions correctly.
   Check existing dtype handling — it may not cover all packed int4
   dtype variants.

3. **Enable routing** (`compressed_tensors.py`): Update the HIP path
   to return the aiter MoE scheme instead of the Triton fallback.
   This may already be done in the checkpoint.

4. **Verify**: Run `bash /workspace/bench_kimi_w4a16_smoke.sh` to
   confirm CK tile path activates without errors.

5. **Optimize decode latency**: Once CK MoE works with W4A16, optimize
   end-to-end decode throughput. The benchmark measures decode median
   latency (ms) on a fixed workload (TP=8, batch=1, input=8192,
   output=2048).

   Areas to explore:
   - CK MoE kernel configuration and tuning for W4A16 shapes
   - GEMM dispatch for the model's specific expert dimensions
   - Attention backend selection and tuning
   - Memory layout optimization

### Important Notes

- Kimi-K2.5 uses `torch.int32` for packed int4 weights, NOT
  `torch.uint32`. Many code paths only check uint32 — you must handle
  both.
- The CK tile `.so` already has compiled A16W4 kernels (verified via
  `strings module_moe_cktile2stages.so | grep a16w4`). No codegen or
  C++ compilation is needed.
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
