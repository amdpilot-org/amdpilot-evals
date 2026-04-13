# Qwen3-VL Triton Attention Throughput Regression Fix

Fix the triton attention backend throughput regression in SGLang when serving Qwen3-VL-8B-Instruct on AMD MI355X.

## Problem — Triton Attention Regression

SGLang v0.5.9 (ROCm 7.2.0, MI355X) with `--attention-backend triton` shows a **33% throughput regression** vs vLLM on the same workload:

| Backend | Output Throughput (tok/s) | TPOT (ms) | E2E Latency p50 (ms) | TTFT p50 (ms) |
|---------|--------------------------|-----------|----------------------|---------------|
| SGLang triton | 1235.85 | 12.21 | 25786 | 1134 |
| SGLang OAI chat | 1115.81 | 13.29 | 29223 | 1267 |
| **vLLM (target)** | **1648.09** | **9.09** | **19385** | **1275** |

The regression is concentrated in **decode throughput**: TPOT 12.21ms (SGLang triton) vs 9.09ms (vLLM) — a 34% gap. Prefill (TTFT ~1100-1275ms) is comparable.

## CRITICAL CONSTRAINTS

1. **The benchmark uses `--attention-backend triton`. This is LOCKED and cannot be changed.** The regression is specifically in the triton attention path on AMD. Switching to the aiter backend is NOT an acceptable fix — it bypasses the regression without fixing it.

2. **Do NOT modify the benchmark script** (`bench_qwen_vl.sh`) or its parameters.

3. **Do NOT set `ATTENTION_BACKEND` in `bench_config.env`.** The benchmark ignores this variable — the backend is hardcoded to triton.

4. The fix must be source-level changes to SGLang's triton attention kernels, model code, scheduler, or memory management — changes that make the triton path faster.

## Deliverable

**A clean git commit on a `fix/qwen-vl-throughput` branch in `/workspace/sglang-fork/`** containing source-level changes that bring triton-backend throughput to ≥1600 tok/s.

## Environment

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — on `sys.path`, used by `python3 -m sglang.*`. **Edit files HERE to modify SGLang behavior.**
- **SGLang fork checkout**: `/workspace/sglang-fork/` — clone of `github.com/Arist12/sglang` for creating the fix branch after optimization.
- **Model weights**: `Qwen/Qwen3-VL-8B-Instruct` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_qwen_vl.sh` — starts server with triton backend, runs warmup + benchmark, reports output throughput.

## Benchmark

The benchmark runs a full serving workload (self-contained):
1. Starts SGLang server with `--attention-backend triton` (hardcoded)
2. Waits for server health
3. Runs 128 image-prompt requests as warmup (results discarded)
4. Runs 128 image-prompt requests as the actual measurement
5. Reports: `Output throughput (tok/s): <value> | concurrency=16 model=Qwen3-VL-8B`

First run takes 15–25 minutes (model loading + CUDA graph compilation + warmup + benchmark). Set `timeout: 2400` or higher when running it. Do NOT use timeout < 2000.

### Quick iteration

For faster feedback during development, you can start the server manually and run `bench_serving` directly with fewer prompts:

```bash
# Start server (background) — MUST use triton
SGLANG_DISABLE_CUDNN_CHECK=1 /opt/venv/bin/python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 --port 30000 --trust-remote-code \
    --attention-backend triton &

# Quick test with 32 prompts
/opt/venv/bin/python3 -m sglang.bench_serving --backend sglang \
    --model Qwen/Qwen3-VL-8B-Instruct --dataset-name image \
    --num-prompts 32 --random-input-len 4000 --random-output-len 2000 \
    --random-range-ratio 1.0 --image-count 1 --image-resolution 720p \
    --image-content random --max-concurrency 16 --seed 123 --warmup-requests 0
```

Use the full benchmark script (`bash /workspace/bench_qwen_vl.sh`) for the final measurement.

## Target

Bring triton-backend output throughput to **≥1600 tok/s** (vLLM achieves 1648 tok/s on the identical workload with its own attention implementation). Current SGLang triton baseline is ~1235 tok/s — this is a 30% gap to close.

## Suggested Investigation Areas

The decode path is where the regression lives (TPOT 12.21ms vs 9.09ms). Focus here:

1. **Triton attention kernel tuning**: The triton decode attention kernels may have suboptimal tile sizes, num_kv_splits, or memory access patterns for MI355X. Profile with `rocprof` to find the hotspot kernels. Check `triton_attention_num_kv_splits` and `triton_attention_reduce_in_fp32` server args.

2. **CUDA graph overhead**: VL models may have variable input shapes that cause excessive CUDA graph recompilation or suboptimal graph capture. Investigate how CUDA graphs interact with the triton attention path.

3. **Vision token handling in decode**: During batched decode with image tokens, the KV cache access pattern may differ from text-only. Check if the triton attention kernel handles multimodal KV entries efficiently.

4. **Scheduling inefficiency**: Image requests with many vision tokens may cause the scheduler to create suboptimal batches during decode.

5. **Memory fragmentation**: Multimodal tokens may cause memory fragmentation in the KV cache, leading to non-contiguous memory access during triton attention.

6. **torch.compile and triton interaction**: Check if `torch.compile` or piecewise CUDA graph compilation introduces overhead specific to the triton attention path on AMD.

## Creating the Fix Branch (REQUIRED)

After achieving the target throughput WITH triton, you **MUST** create a clean git commit:

```bash
# 1. Identify changes in the runtime
diff -ruN /workspace/sglang-fork/python/sglang/ /sgl-workspace/sglang/python/sglang/ > /workspace/changes.patch

# 2. Create the fix branch
cd /workspace/sglang-fork
git checkout -b fix/qwen-vl-throughput

# 3. Apply and commit
git apply /workspace/changes.patch || patch -p0 < /workspace/changes.patch
git add -A
git commit -m "fix: optimize triton attention for Qwen3-VL on MI355X

Closes the ~33% decode throughput gap vs vLLM on bench_serving image workload.
Baseline: ~1235 tok/s -> Target: >=1600 tok/s (with --attention-backend triton)

Changes:
- <describe what you changed and why>
"
```

If the diff approach fails, manually copy each changed file:
```bash
cd /workspace/sglang-fork
git checkout -b fix/qwen-vl-throughput
cp /sgl-workspace/sglang/python/sglang/<changed_file> python/sglang/<changed_file>
git add -A
git commit -m "fix: optimize triton attention for Qwen3-VL on MI355X"
```

## Rules

- Do NOT use `pkill -f` — it kills your own shell. Use targeted `kill <PID>`.
- Read error messages carefully and fix the root cause.
- **The benchmark uses `--attention-backend triton`. Do NOT switch to aiter or any other backend.** The goal is to fix triton performance.
- Do NOT modify the benchmark script or its parameters.
- **Run `bash /workspace/bench_qwen_vl.sh` as your LAST command before creating the git branch.**
- **Create the fix branch as the very last step.**
- Kill leftover sglang server processes before starting a new one:
  `ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9`
