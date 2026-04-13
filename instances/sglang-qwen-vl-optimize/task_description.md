# Qwen3-VL Triton Attention Throughput Regression

## Problem

SGLang v0.5.9 (ROCm 7.2.0, MI355X) with `--attention-backend triton` shows a **33% throughput regression** vs vLLM on the same image-serving workload:

| Backend | Output Throughput (tok/s) | TPOT (ms) |
|---------|--------------------------|-----------|
| SGLang triton | ~1235 | 12.2 |
| vLLM (reference) | ~1648 | 9.1 |

The regression is concentrated in **decode throughput** (TPOT). Prefill performance is comparable.

## Constraints

- The benchmark uses `--attention-backend triton`. This is locked and cannot be changed. Switching backends is not an acceptable fix.
- Do NOT modify the benchmark script or its parameters.
- The fix must be source-level changes that make the triton attention path faster.

## Environment

- **SGLang runtime**: `/sgl-workspace/sglang/` — edit files here to modify runtime behavior.
- **Model weights**: `Qwen/Qwen3-VL-8B-Instruct` cached at `/root/.cache/huggingface`.
- **Benchmark**: `bash /workspace/bench_qwen_vl.sh` — runs a full serving workload and reports output throughput.

First run takes 15-25 minutes (model loading + CUDA graph compilation + warmup). Use `timeout: 2400` or higher.

## Rules

- Do NOT use `pkill -f`. Use targeted `kill <PID>`.
- Read error messages carefully and fix the root cause.
- Run `bash /workspace/bench_qwen_vl.sh` as your last command.
