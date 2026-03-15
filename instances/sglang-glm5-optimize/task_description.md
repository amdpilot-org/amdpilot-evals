# GLM-5-FP8 Decode Latency Optimization

Optimize the decode latency of the GLM-5-FP8 model on 8× AMD MI355X GPUs using SGLang.

## Environment (read carefully)

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — this is on `sys.path` and is what `python3 -m sglang.*` uses. **Edit files HERE to modify SGLang behavior.**
- **SGLang reference checkout**: `/workspace/sglang/` — a fresh `git clone` for reference only. Changes here do NOT affect the runtime.
- **Model weights**: `zai-org/GLM-5-FP8` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_glm5.sh` — pre-built and working. Run it first to establish a baseline.

## Step 1 — Establish Baseline (do this FIRST)

```bash
bash /workspace/bench_glm5.sh
```
This prints `Decode median (ms): <value> | tp=8 batch=1`. Update optimization_state.json.

## Step 2 — Profile to Find Bottlenecks

Use `torch.profiler` or `rpd` to identify the top GPU kernels by time. Example:
```python
import torch
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # run a few decode steps
    pass
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```
Classify bottlenecks: attention kernels, MoE dispatch/fused_moe, all-reduce, GEMM, other.

## Step 3 — Source-Level Optimizations (REQUIRED)

Config tuning alone (env vars, mem-fraction) yields marginal gains. You MUST attempt
source-level changes in `/sgl-workspace/sglang/`. Concrete targets:

1. **Attention backend** (`/sgl-workspace/sglang/python/sglang/srt/layers/attention/`):
   - Check the tilelang NSA decode kernel for unnecessary synchronization
   - Look at `nsa_backend.py` `_forward_tilelang` — can the tiling be improved?

2. **MoE kernels** (`/sgl-workspace/sglang/python/sglang/srt/layers/moe/`):
   - Profile `fused_moe` dispatch — is expert routing efficient?
   - Check if `aiter` fused_moe is being used (preferred on AMD)

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
