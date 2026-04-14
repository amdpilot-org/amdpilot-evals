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

## Checkpoint State

Previous trials completed several integration steps that are already in
the container. The W4A16 kernel infrastructure (codegen, scheme class,
weight loading, imports, dtype detection) is in place. However, the MoE
routing path on HIP does not yet use the CK backend for this weight format.

## Known Constraints

- This model uses ~273GB VRAM on 8×288GB GPUs. Any approach that
  creates temporary full-precision copies of all expert weights will OOM.
- CUDA graphs must remain enabled — approaches that deadlock during
  graph capture are not viable.
- Kimi-K2.5 uses `torch.int32` for packed int4 weights, NOT
  `torch.uint32`. Code paths may only check one — handle both.

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
