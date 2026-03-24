# Kimi-K2.5 Decode Latency Optimization

Optimize the decode latency of the Kimi-K2.5 (1T MoE) model on 8x AMD MI355X GPUs using SGLang.
The single metric that matters is the output of `/workspace/bench_kimi_k25.sh`. Lower is better.

## Environment

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — on `sys.path`, used by `python3 -m sglang.*`. **Edit files HERE to modify SGLang behavior.**
- **SGLang reference checkout**: `/workspace/sglang/` — fresh `git clone` for reference only. Changes here do NOT affect the runtime.
- **AITER library**: `/sgl-workspace/aiter/` — AMD inference acceleration library. May need modifications for Kimi-K2.5 support.
- **Model weights**: `moonshotai/Kimi-K2.5` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_kimi_k25.sh` — runs `sglang.bench_one_batch` with fixed workload params.

## Model Architecture

Kimi-K2.5 is a 1T-parameter MoE model (DeepSeek V3 architecture):
- 384 experts, 8 active per token, 1 shared expert
- MLA (Multi-head Latent Attention) with 64 attention heads
- hidden_size=7168, MoE hidden dim per expert=2048
- 61 layers (including 1 dense layer), SwiGLU activation
- Requires `--trust-remote-code`

## Benchmark

The benchmark runs `sglang.bench_one_batch` with a long-context workload and prints:
  `Decode median (ms): <value> | tp=8 batch=1 in=8192 out=2048 decode=<backend>`

Fixed parameters: tp=8, batch_size=1, input_len=8192, output_len=2048.
First run takes a long time (model loading + CUDA graph compilation).
Set `timeout: 1200` when running it. If it times out, kill leftover sglang processes
(`ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9`) before retrying.

### Backend configuration

The benchmark defaults to `triton` decode / `aiter` prefill attention backends.
To change backends, write to `/workspace/bench_config.env`:
```bash
export DECODE_ATTENTION_BACKEND=aiter
export PREFILL_ATTENTION_BACKEND=aiter
export SGLANG_ROCM_FUSED_DECODE_MLA=1
```
This ensures the verification run uses the same configuration as your run.

Read the benchmark output logs carefully to identify which backends are active (attention,
MoE, all-reduce) before optimizing. Only optimize backends that are actually in use.

## Optimization Objective

Your single goal: **minimize the decode median latency reported by `bench_kimi_k25.sh`.**

There are NO restrictions on approach. You are free to:
- Switch between any available attention backends (triton, aiter, flashinfer, or any combination)
- Tune MoE kernel configurations (GEMM block sizes, waves_per_eu, etc.)
- Tune GEMM configs for small-M shapes (batch=1 decode)
- Fix bugs in any backend to unlock better performance
- Modify source code in `/sgl-workspace/sglang/` and `/sgl-workspace/aiter/`
- Try any optimization technique: kernel config tuning, torch.compile, graph capture, scheduling, backend switching, etc.

The only constraint: the final metric must come from `bench_kimi_k25.sh` with CUDA graphs enabled (no `--disable-cuda-graph`).

## Deliver Results

After optimization is complete, push your changes and write a setup guide:

1. **Create branches**: The fork remotes are pre-configured:
   - `/sgl-workspace/sglang/` has remote `fork` → `git@github.com:Arist12/sglang.git`
   - `/sgl-workspace/aiter/` has remote `fork` → `git@github.com:Arist12/aiter.git`
   - SSH keys are mounted at `/root/.ssh/`
   - Set `GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no'` before git operations

2. **Push changes**: Create branch `kimi-k25-optimize-v3` on each fork with your changes:
   ```bash
   cd /sgl-workspace/sglang
   git checkout -b kimi-k25-optimize-v3
   git add -A && git commit -m "Kimi-K2.5 decode optimization for MI355X"
   GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git push fork kimi-k25-optimize-v3
   ```
   Do the same for aiter if you modified it.

3. **Write setup guide**: Create `/workspace/setup_guide.md` with:
   - Step-by-step reproduction instructions starting from the base Docker image
   - Which branches to clone and how to set up the environment
   - The exact benchmark command and expected results
   - Summary of all optimizations applied and their individual impact

4. **Verify**: Clone your branches into a clean directory and confirm the benchmark runs correctly.

## Rules

- Do NOT use `pkill -f` — it kills your own shell. Use targeted `kill <PID>`.
- Read error messages carefully and fix the root cause.
- Final metrics must use CUDA graphs (no `--disable-cuda-graph`).
- Run `bench_kimi_k25.sh` as your LAST command.
