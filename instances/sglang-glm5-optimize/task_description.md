# GLM-5.1-FP8 Decode Latency Optimization

Optimize the decode latency of the GLM-5.1-FP8 model on 8× AMD MI355X GPUs using SGLang.

## Environment

- **Installed SGLang runtime**: `/sgl-workspace/sglang/` — on `sys.path`, used by `python3 -m sglang.*`. **Edit files HERE to modify SGLang behavior.**
- **SGLang reference checkout**: `/workspace/sglang/` — fresh `git clone` for reference only. Changes here do NOT affect the runtime.
- **Model weights**: `zai-org/GLM-5.1-FP8` cached at `/root/.cache/huggingface`.
- **Benchmark script**: `/workspace/bench_glm5.sh` — runs `sglang.bench_one_batch` with fixed workload params.

## Benchmark

The benchmark runs `sglang.bench_one_batch` and prints:
  `Decode median (ms): <value> | tp=8 batch=1`

First run takes ~5 minutes (model loading from local NVMe + CUDA graph compilation).
Set `timeout: 600` when running it. If it times out, kill leftover sglang processes
(`ps aux | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9`) before retrying.

`bench_one_batch` supports backend selection flags such as `--attention-backend`,
`--decode-attention-backend`, `--fp8-gemm-backend`, etc. The benchmark script uses
defaults but sources `/workspace/bench_config.env` if it exists. To set environment
variables that configure backends, write them to that file:
```bash
echo 'export SGLANG_ATTENTION_BACKEND=aiter' > /workspace/bench_config.env
```
This ensures the verification run uses the same configuration as your run.

Read the benchmark output logs carefully to identify which backends are active (attention,
MoE, all-reduce) before optimizing. Only optimize backends that are actually in use.

## Deliver Results

After optimization is complete, push your changes and write a setup guide:

1. **Create branches**: The fork remotes are pre-configured:
   - `/sgl-workspace/sglang/` has remote `fork` → `git@github.com:Arist12/sglang.git`
   - `/sgl-workspace/aiter/` has remote `fork` → `git@github.com:Arist12/aiter.git`
   - SSH keys are mounted at `/root/.ssh/`
   - Set `GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no'` before git operations

2. **Push changes**: Create branch `glm5-optimize` on each fork with your changes:
   ```bash
   cd /sgl-workspace/sglang
   git checkout -b glm5-optimize
   git add -A && git commit -m "GLM-5.1-FP8 decode optimization for MI355X"
   GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git push fork glm5-optimize
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
- Run `bench_glm5.sh` as your LAST command.
