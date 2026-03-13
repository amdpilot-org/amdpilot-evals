# GLM-5-FP8 Decode Latency Optimization

Optimize the decode latency of the GLM-5-FP8 model on 8× AMD MI355X GPUs using SGLang.

## Environment (read carefully)

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — this is on `sys.path` and is what `python3 -m sglang.*` uses. **Edit files HERE to modify SGLang behavior.**
- **SGLang reference checkout**: `/workspace/sglang/` — a fresh `git clone` for reference only. Changes here do NOT affect the runtime.
- **Model weights**: `zai-org/GLM-5-FP8` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_glm5.sh` — pre-built and working. Run it first to establish a baseline.

## Step 1 — Establish Baseline (do this FIRST)

**NOTE**: The benchmark loads a large TP=8 model and takes 10-15 minutes on first run.
Your shell timeout is 300 seconds. Run the benchmark like this:

```bash
# Run in background, then poll for completion:
nohup bash /workspace/bench_glm5.sh > /tmp/bench_output.txt 2>&1 &
BENCH_PID=$!
# Wait and check periodically:
sleep 300 && tail -20 /tmp/bench_output.txt
# If not done yet, wait more:
sleep 300 && tail -20 /tmp/bench_output.txt
```

Alternatively, just run `bash /workspace/bench_glm5.sh` — if it times out at 300s,
the server stays running in the background. Run it AGAIN and it will connect to the
already-loaded server and complete instantly.

This prints `Decode median (ms): <value> | tp=8 batch=1`. Update optimization_state.json.

**IMPORTANT**: The benchmark uses SGLang auto-detection for attention backend, MoE
implementation, and all-reduce. Read the benchmark output logs carefully to determine
which backends are ACTUALLY active before optimizing. Look for lines like:
- `Use nsa attention backend` / `Use flashinfer attention backend`
- `NSA backends: prefill=..., decode=...`
- `[AR] Using AiterCustomAllreduce` / `[AR] Using NCCL`
- `[aiter] [fused_moe] using 1stage default` / `using Triton fused_moe`

Only optimize the backends that are actually in use.

## Step 2 — Profile to Find Bottlenecks

Use `torch.profiler` to identify the top GPU kernels by time. **CUDA graphs hide
individual kernel timings** — profile with `--disable-cuda-graph` to see the real
breakdown, then apply optimizations and measure with CUDA graphs re-enabled.

```python
# Example: profile SGLang decode steps
import torch
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # run a few decode steps
    pass
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```
Classify bottlenecks: attention kernels, MoE dispatch/fused_moe, all-reduce, GEMM, other.
Focus optimization on the kernel category that takes the most GPU time.

## Step 3 — Source-Level Optimizations (REQUIRED)

Config tuning alone (env vars, mem-fraction) yields marginal gains. You MUST attempt
source-level changes in `/sgl-workspace/sglang/`. First check the benchmark logs to
determine which backends are active, then target those specific implementations.

Potential targets (check benchmark logs to confirm which are active):

1. **Attention backend** (`/sgl-workspace/sglang/python/sglang/srt/layers/attention/`):
   - Check which backend is active (tilelang, aiter, flashmla, etc.)
   - For tilelang: check NSA decode kernel tiling and synchronization

2. **MoE kernels** (`/sgl-workspace/sglang/python/sglang/srt/layers/moe/`):
   - Check if aiter or Triton fused_moe is active (check benchmark logs)
   - Profile fused_moe dispatch — is expert routing efficient?

3. **All-reduce** (`/sgl-workspace/sglang/python/sglang/srt/layers/`):
   - With TP=8, all-reduce is on the critical path
   - Check if custom all-reduce (AiterCustomAllReduce) is active

4. **CUDA graph capture** — check if all decode layers are captured in the graph.
   Any graph breaks force CPU-GPU synchronization.

5. **Scheduling / batching** — for batch=1 decode, check if there's overhead from
   dynamic batching logic that can be bypassed.

After EACH change, re-run `bash /workspace/bench_glm5.sh` and compare.

## Rules

- Do NOT use `pkill -f` — it kills your own shell. Use targeted `kill <PID>`.
- Read error messages carefully and fix the root cause.
- Final metrics must use CUDA graphs (no `--disable-cuda-graph`).
- Run `bench_glm5.sh` as your LAST command.
- Do NOT modify benchmark parameters (model, tp, batch, input/output lengths).
