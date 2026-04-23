# Optimize Qwen3-30B-A3B MoE Training MFU on MI355X (8-GPU single node)

GitHub issue: https://github.com/amdpilot-org/Primus/issues/1

## Objective

Maximize the training throughput (TFLOP/s/GPU) of Qwen3-30B-A3B MoE
pretraining on a single node of 8x AMD Instinct MI355X GPUs using the
Primus framework. The current baseline is **~300 TFLOP/s/GPU**
(~740ms/iter). Your goal is to push this number as high as possible.

## Environment

- **Base image**: `primus-qwen3-30b-mfu-base:v1` (built from `primus-mi355x-flat:v1` + uv)
- **Primus source**: `/workspace/primus_train/Primus` (branch `mfu-optimization`, commit `e50a78b`)
- **Primus-Turbo source**: `/workspace/primus_train/Primus-Turbo` (commit `3cd482d`)
- **Hardware**: 8x MI355X GPUs (288 GB HBM each), single node
- **ROCm**: 7.0
- **PyTorch**: 2.9.0a0
- **Triton**: 3.4.0
- **Python**: system python3

## Model Configuration (IMMUTABLE)

- **Model**: Qwen3-30B-A3B (MoE, 128 experts, top-8 routing, 48 layers)
- **Parallelism**: TP=1, EP=8, PP=1
- **Batch**: `micro_batch_size=1`, `global_batch_size=8`, `seq_length=8192` (no gradient accumulation)
- **Precision**: BF16
- **Data**: mock data (`--mock_data True`)

## Baseline Optimizations (already enabled)

| Optimization | Status |
|---|---|
| Turbo Attention (flash attention) | ON |
| Turbo Grouped MLP (auto-tuned GEMM backend) | ON |
| Turbo DeepEP (optimized MoE dispatch/combine) | ON |
| Sync-Free MoE Stage 1 (fused router + permute) | ON |
| Fused RoPE | ON |
| Fused Cross-Entropy | ON |
| Gradient Accumulation Fusion | ON |
| Precision-Aware Optimizer (bf16 states) | ON |
| Activation Recompute (5 layers, full/block) | ON |

## Expected Baseline / Reference Setup

- **Reference throughput**: ~300 TFLOP/s/GPU (~740ms/iter)
- **Memory usage**: ~145 GB / 288 GB (50%)
- **Reference commit**: Primus `e50a78b`, Primus-Turbo `3cd482d`
- **Image**: `primus-qwen3-30b-mfu-base:v1`
- **GPU shape**: 8xMI355X
- **Verified baseline from prior run (r36)**: 294.81 TFLOP/s/GPU

Before optimizing, you MUST first reproduce the baseline by running
`bash /workspace/bench_mfu.sh`. The result should be in the range of
~270-330 TFLOP/s/GPU. If the measured baseline is below 270 TFLOP/s/GPU,
treat this as an environment/reproduction mismatch and resolve it
before claiming optimization progress. Do NOT disable or rewrite any
of the baseline-enabled flags listed above before you have reproduced
the reference setup.

## Benchmark

Run the benchmark:

```bash
bash /workspace/bench_mfu.sh
```

This script runs 10 training iterations, discards warmup (iterations
1-2), and reports the average steady-state TFLOP/s/GPU. The script
outputs a line:

```
TFLOPS_PER_GPU: <value>
```

The metric to optimize is **TFLOPS_PER_GPU** — higher is better.

## Prior Observations (from the issue)

- With `mbs=4, gbs=256, seq=4096` (8 grad-accum steps), throughput
  reaches ~400 TFLOP/s/GPU (~7.5s/iter). The higher per-GPU token count
  better saturates compute. However the benchmark config keeps mbs=1.
- Sync-Free MoE Stage 2 and `use_turbo_parallel_linear` both
  **regressed** throughput to ~380 TFLOP/s in the larger batch config.
- TP=2 x EP=4 is ~9x slower (~45 TFLOP/s/GPU) due to TP all-reduce
  overhead.
- Profiler traces are available — run
  `bash scripts/run_qwen3_30b_mfu_baseline.sh --profile` to collect
  them.

## Prior Run Learnings

- **r36** verified baseline at 294.81 TFLOP/s/GPU — this is the proven
  working environment.
- **r21** replaced CK grouped GEMM backend with Triton grouped GEMM in
  Primus-Turbo permutation layer, and enabled fused activation with
  probabilities.

## Optimization Ideas to Explore

- Analyze baseline profiler trace to identify top bottlenecks
- Investigate why mbs=1/seq=8192 underutilizes compute vs mbs=4/seq=4096
- Profile and optimize MoE GEMM kernel selection at this batch size
- Investigate communication overlap opportunities
- Explore activation recompute tuning (currently 5 layers — try
  different values)
- Look for kernel fusion opportunities in the MoE path
- Consider memory headroom: only 50% used — can trade memory for
  compute

## Workspace Layout

- Primus repo: `/workspace/primus_train/Primus/`
- Primus-Turbo: `/workspace/primus_train/Primus-Turbo/`
- Benchmark script: `/workspace/bench_mfu.sh`
- Baseline training script: `/workspace/primus_train/Primus/scripts/run_qwen3_30b_mfu_baseline.sh`
- Model config: `/workspace/primus_train/Primus/examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml`

## Rules

- Edit files under `/workspace/primus_train/Primus/` or
  `/workspace/primus_train/Primus-Turbo/` — changes take effect
  immediately
- Do NOT run `pip install -e .` on Primus-Turbo unless you have changed
  its C++ extensions — this triggers a long rebuild
- Do NOT modify the benchmark script `/workspace/bench_mfu.sh`
- Do NOT add debug print statements — make clean, minimal changes only
- Do NOT change the model architecture (number of experts, layers,
  hidden size, etc.)
- Do NOT change global_batch_size, micro_batch_size, or seq_length —
  the benchmark config is fixed
- You MAY change parallelism strategies, kernel backends, fusion flags,
  recompute settings, environment variables, and other training
  hyperparameters that don't alter the model or data shape
- Before running any benchmark: kill leftover training processes first:
  `pgrep -f 'python3 -m primus' | xargs -r kill -9; sleep 2`
  NEVER use `pkill -f primus` — this may match the agent shell and
  self-kill.
- Collect profiler traces if needed to understand bottlenecks:
  `bash scripts/run_qwen3_30b_mfu_baseline.sh --profile`
- `git diff HEAD` in `/workspace/primus_train/Primus/` will capture all
  your changes for the final patch
