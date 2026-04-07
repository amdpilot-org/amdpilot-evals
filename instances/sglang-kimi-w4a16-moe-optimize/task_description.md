# Kimi K2.5 W4A16 MoE Support and Optimization on ROCm

## Context

Kimi K2.5 is a 1T-parameter MoE model (DeepSeek V3 architecture) with 384
experts (8 active per token), using INT4 weight-only quantization (W4A16)
via the `compressed-tensors` format. It is one of the largest publicly
available MoE models and a key target for AMD MI355X inference.

## Current State

The model **fails to load** on the current ROCm/SGLang installation. During
MoE layer initialization, the quantized MoE dispatch path selects a kernel
backend that is not available on HIP, causing a crash before inference can
begin.

Errors indicate missing or incompatible kernel support for the W4A16 MoE
dispatch path on this platform.

## Task

1. **Make the model serve**: Fix the MoE layer initialization so the model
   loads and runs inference on ROCm without crashing. The W4A16 quantized
   weight format (int32-packed INT4) must be correctly handled through the
   dispatch, weight conversion, and forward pass.

2. **Optimize decode latency**: Once the model is serving, optimize the
   decode throughput. The benchmark measures decode median latency (ms) on
   a fixed workload (TP=8, batch=1, input=8192, output=2048).

   Areas to explore:
   - MoE kernel backend selection and tuning
   - GEMM configuration for the model's specific shapes
   - Attention backend selection (triton vs aiter)
   - CUDA graph compatibility
   - Memory layout and allocation

## Environment

- Repository: SGLang (editable install at `/sgl-workspace/sglang/`)
- AITER library at `/sgl-workspace/aiter/`
- Model weights cached at `/root/.cache/huggingface`
- Docker container with ROCm, PyTorch, 8x AMD MI355X GPUs
- Use `/opt/venv/bin/python3` for all commands

## Benchmark

The benchmark script at `/workspace/bench_kimi_w4a16.sh` runs
`sglang.bench_one_batch` with fixed parameters. Do NOT modify the benchmark
parameters (model, tp, batch size, input/output lengths).

You may write `/workspace/bench_config.env` to configure environment
variables (e.g., backend selection flags). This file is sourced by the
benchmark script before each run.

```bash
bash /workspace/bench_kimi_w4a16.sh
```

The benchmark outputs `Decode median (ms): <value>` — lower is better.

## Rules

- Do not use `pkill -f` to kill processes
- CUDA graphs must remain enabled (do not disable them)
- Run `bash /workspace/bench_kimi_w4a16.sh` as the last command to verify
  your changes
