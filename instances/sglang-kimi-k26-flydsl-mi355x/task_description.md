# Kimi-K2.6 MI355X FlyDSL fused-MoE optimization

GitHub issue: <https://github.com/amdpilot-org/sglang/issues/2>
Baseline PR (source-of-truth): <https://github.com/sgl-project/sglang/pull/23381>
Reference blog (MI300X K2.5 precedent): <https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html>

## Objective

Port the FlyDSL-based fused-MoE kernel stack from AMD's Kimi-K2.5 MI300X
blog to **Kimi-K2.6 on 4x MI355X**, on top of the baseline established by
[PR #23381](https://github.com/sgl-project/sglang/pull/23381). Success is
defined by a measurable, reproducible improvement on the
`concurrency=40` decode-dominated serving workload — where the MoE
kernel is the dominant bottleneck (88-90% of GPU time per blog) — while
**not regressing** the `BS=1` decode path that PR #23381 already tuned.

The amdpilot executor (Kimi-K2.6 in ralph mode) iterates autonomously
through the phased plan below. Phase 1 (Opus agent) first reproduces the
baseline on this image and commits `:phase1-baseline` so every trial
starts from a verified-identical snapshot.

## Key gating env vars and guards (supervisor read this)

- **FlyDSL gates** (knobs, NOT scored tunables): `AITER_USE_FLYDSL_MOE`,
  `AITER_USE_FLYDSL_MOE_STAGE1`, `AITER_USE_FLYDSL_MOE_STAGE2`,
  `FLYDSL_W4A16_HYBRID`, `AITER_ENFORCE_DSL=1`.
- **Container paths** (immutable): `DSL2_ROOT=/opt/FlyDSL`,
  `MLIR_PATH=/opt/mlir_install`, AITER pinned to `dev/kimi-K2.5`.
- **Primary metric**: `output_throughput_tok_s` at concurrency=40,
  target **>= 1.30x** Phase 1 baseline.
- **Guard metric**: `decode_bs1_in8k` at BS=1 input=8192, target
  **>= 0.98x** Phase 1 baseline (PR #23381 published ~38.05 tok/s).
- **Accuracy gate**: GSM8K `exact_match_flexible` @ limit=50 **>= 0.90**.
- **Phases**: A import on gfx950 -> B stage1 -> C stage1+stage2 ->
  D `FLYDSL_W4A16_HYBRID` sweep -> E `--enable-torch-compile` ->
  F `--disable-radix-cache` (random-input only) -> G GSM8K gate.
- **Scored-trial rule**: every scored optimization trial requires a
  tracked source edit in AITER (`dev/kimi-K2.5`,
  `aiter/ops/fused_moe/flydsl_moe_stage{1,2}.py`) or in the FlyDSL kernel
  at `/opt/FlyDSL/python/kernels/mixed_moe_gemm_2stage.py`. Env-var-only
  toggles do NOT count and will be rejected by the supervisor.
- **Silent-fallback guard**: always run with `AITER_ENFORCE_DSL=1` so a
  FlyDSL import/compile failure is a hard error rather than a silent
  fallback to Triton. A silent-fallback win is not a real win.
- **Immutables**: TP=4, `--mem-fraction-static 0.85`,
  `--context-length 128000`, `bench_flydsl_k26.sh`, `test_harness.py`,
  PR #23381's tuned Triton MoE JSON config (FlyDSL replaces Triton at
  runtime via env vars; the Triton config remains the fallback per
  batch-size bucket — do NOT delete it).

## Environment

- **Base image**: `jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420`
  (public Docker Hub, built from PR #23381)
- **Layered on top**: FlyDSL `main` at `/opt/FlyDSL`, AITER
  `dev/kimi-K2.5` branch at `/sgl-workspace/aiter`
- **SGLang**: editable install at `/sgl-workspace/sglang` (PR #23381
  patches already applied inside the base image)
- **Hardware**: 4x AMD Instinct MI355X (288 GB HBM3E each), TP=4
- **ROCm**: 7.2.0
- **SGLang**: v0.5.10rc0
- **Triton**: 3.6.0
- **Python**: `/opt/venv/bin/python3`
- **Model weights**: `/sgl-workspace/models/Kimi-K2.6` (baked into image)

## Baseline (from PR #23381, unmodified)

Measured on 4x MI355X (GPUs 0-3), TP=4, `bench_one_batch_server`,
`--batch-size 1 --output-len 1024 --skip-warmup`:

| input_len | TTFT (s) | Decode (tok/s) | Overall (tok/s) | Total latency (s) |
| --------- | -------- | -------------- | --------------- | ----------------- |
| 1,024     | 0.44     | 40.24          | 80.47           | 25.45             |
| 2,048     | 0.44     | 40.01          | 120.04          | 25.59             |
| 4,096     | 0.66     | 39.07          | 195.33          | 26.21             |
| 8,192     | 0.70     | 38.05          | 342.44          | 26.91             |
| 16,384    | 1.17     | 35.85          | 609.49          | 28.56             |
| 32,768    | 2.42     | 32.03          | 1056.88         | 31.97             |

Phase 1 on this image will:
1. Run `bash /workspace/bench_flydsl_k26.sh` **with FlyDSL disabled**
2. Record `output_throughput_tok_s` at concurrency=40 (expected baseline
   value is NOT in the PR; phase1 pins it)
3. Verify `decode_bs1_in8k` reproduces **~38.05 tok/s** (within 2% of
   PR #23381's published value for input=8192)
4. Commit `:phase1-baseline`, fill `baseline_contract.expected_metric`,
   hand off to executor trials

## Target

Primary metric (`output_throughput_tok_s`, higher is better):

- **Must**: `output_throughput_tok_s` @ concurrency=40 >=
  **1.30x phase1 baseline**. Conservative vs the blog's +162% on MI300X,
  because MI355X shapes and the K2.6 W4A16 path may exhibit different
  kernel-selection behavior.

Guard metric (`decode_bs1_in8k`, reported in the same bench line):

- **Must**: `decode_bs1_in8k` >= **0.98 x phase1 baseline**. Any
  FlyDSL path that regresses the PR #23381 BS=1 number is rejected
  even if it wins at concurrency=40.

Accuracy gate (run via `test_harness.py` in Phase G only):

- **Must**: `lm_eval` GSM8K `exact_match_flexible` @ limit=50 >=
  **0.90** (blog reports 0.96 @ limit=100).

## Phased plan

Each phase is one or more trials. The executor explores deeper phases
only after the previous phase's best config is measured and recorded.

### Phase A - FlyDSL loadability on K2.6 weights

- Verify `import kernels.moe_gemm_2stage` works under
  `/opt/FlyDSL/python/` (after `PYTHONPATH=/opt/FlyDSL/python python3`)
- Verify `kernels/mixed_moe_gemm_2stage.py` compiles on `gfx950`
  (MI355X). FlyDSL README claims gfx950 support; verify in practice.
- Verify AITER's `flydsl_moe_stage1` / `flydsl_moe_stage2` hooks see
  `DSL2_ROOT=/opt/FlyDSL` in `sys.path`.
- If Phase A fails (kernel compile error for gfx950, import error, or
  AITER dispatch mismatch on K2.6 W4A16 tensors): record failure mode
  and fall through to framework-only optimizations (Phase E + F).

### Phase B - FlyDSL Stage 1 only (gate/up projection)

```bash
# /workspace/bench_config.env
export AITER_USE_FLYDSL_MOE=1
export AITER_USE_FLYDSL_MOE_STAGE1=1
export AITER_USE_FLYDSL_MOE_STAGE2=0
export AITER_ENFORCE_DSL=1
```

Measure both benchmarks. Expected: modest win from replacing Triton
gate/up projection with FlyDSL while keeping Triton down-projection.

### Phase C - FlyDSL Stage 1 + Stage 2

```bash
export AITER_USE_FLYDSL_MOE_STAGE2=1
```

The blog's biggest gains come from the full two-stage replacement.

### Phase D - Mixed-precision hybrid sweep

```bash
# Option 1: W4A16 on both stages (default when FLYDSL_W4A16_HYBRID unset)
unset FLYDSL_W4A16_HYBRID

# Option 2: W4A16 Stage1, BF16 Stage2 (blog default)
export FLYDSL_W4A16_HYBRID=w2_bf16
```

Compare both. Blog notes `w2_bf16` trades memory for numerical stability
on Stage 2.

### Phase E - torch.compile (framework-level, MoE-orthogonal)

```bash
export EXTRA_SERVER_FLAGS="${EXTRA_SERVER_FLAGS} --enable-torch-compile"
```

Blog reports this is especially effective at concurrency=40 because it
removes per-kernel dispatch overhead in decode.

### Phase F - Disable radix cache (random-input workload only)

```bash
export EXTRA_SERVER_FLAGS="${EXTRA_SERVER_FLAGS} --disable-radix-cache"
```

Our benchmark uses `--dataset-name random --random-range-ratio 1.0`, so
there is no shared prefix for the radix tree to exploit. Disabling
frees memory for KV cache.

### Phase G - Best-config accuracy validation

After the executor converges on a best `bench_config.env` + server
flag combination:

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```

Runs `lm_eval` GSM8K limit=50. Must report `SCORE: 100` (i.e.
`exact_match_flexible >= 0.90`). If it fails, the FlyDSL config is
rejected regardless of throughput win.

## Environment variables reference (from blog)

| Variable                          | Value                             | Purpose                                                             |
| --------------------------------- | --------------------------------- | ------------------------------------------------------------------- |
| `AITER_USE_FLYDSL_MOE`            | `1` (enable), `0` (baseline)      | Master gate for the FlyDSL MoE path inside AITER's fused_moe.       |
| `AITER_USE_FLYDSL_MOE_STAGE1`     | `1` to replace gate/up projection | Stage 1 of the two-stage MoE.                                       |
| `AITER_USE_FLYDSL_MOE_STAGE2`     | `1` to replace down projection    | Stage 2 of the two-stage MoE.                                       |
| `AITER_ENFORCE_DSL`               | `1`                               | Error if FlyDSL is requested but unavailable (avoids silent fallback). |
| `DSL2_ROOT`                       | `/opt/FlyDSL`                     | Where AITER adds FlyDSL to `sys.path`. Set in Dockerfile.           |
| `MLIR_PATH`                       | `/opt/mlir_install`               | FlyDSL compiler dependency. Set in Dockerfile.                      |
| `FLYDSL_W4A16_HYBRID`             | unset or `w2_bf16`                | `w2_bf16` keeps Stage1 W4A16 and runs Stage2 in BF16.               |
| `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT` | `3`                             | Blog-recommended CK tile conversion mode.                           |
| `TRITON_MAX_CACHE_SIZE`           | `2147483648`                      | 2 GiB Triton kernel cache ceiling.                                  |
| `AITER_FLYDSL_MOE_COMPARE`        | `0`                               | Disable per-call correctness comparison (debug only).               |
| `AITER_FLYDSL_MOE_COMPARE_STAGE2` | `0`                               | Same, Stage 2.                                                      |
| `AITER_FLYDSL_DEBUG`              | `0`                               | Silence debug logging.                                              |
| `SGLANG_USE_AITER`                | `1`                               | Route SGLang MoE through AITER (enables FlyDSL hook).               |
| `SGLANG_ROCM_FUSED_DECODE_MLA`    | `0`                               | PR #23381 immutable; `1` crashes triton decode backend.             |
| `SGLANG_DEEPSEEK_LOAD_MAX_WORKERS` | `4`                              | PR #23381: bound DeepSeek-loader threadpool.                        |

## Immutables (do NOT modify)

- Model path: `/sgl-workspace/models/Kimi-K2.6`
- TP size: 4 (4x MI355X, GPUs 0-3)
- Attention backends: `--decode-attention-backend triton --prefill-attention-backend aiter`
- `--mem-fraction-static 0.85`
- `--context-length 128000`
- `--disable-custom-all-reduce`
- `/workspace/bench_flydsl_k26.sh` (immutable artifact)
- `/workspace/test_harness.py` (immutable artifact)
- PR #23381's tuned Triton MoE JSON config:
  `/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json`
  (FlyDSL should **replace Triton at runtime** via env vars, not by
  deleting the config file — the config remains the fallback when
  FlyDSL is disabled on any given batch-size bucket.)

## Rules

- Use `/opt/venv/bin/python3` for every Python invocation. The container
  has no system `pip`.
- Do NOT run `pip install -e .` on `/sgl-workspace/sglang` or
  `/sgl-workspace/aiter` — they are already editable-installed and a
  reinstall triggers a long rebuild.
- Do NOT use `pkill -f python` or similar; it can match the amdpilot
  runtime shell and exit 137 the trial. Use `kill <pid>` or the
  benchmark script's own `trap cleanup EXIT` path.
- CUDA graphs must remain enabled (the PR image captures them at
  startup; do not pass `--disable-cuda-graph`).
- `bash /workspace/bench_flydsl_k26.sh` is the last command before any
  verification run; its output line
  `output_throughput_tok_s: <v> | concurrency=40 in=10240 out=512 decode_bs1_in8k=<g>`
  is the ONLY metric the orchestrator scrapes.
- `git diff HEAD` in `/sgl-workspace/sglang` and `/sgl-workspace/aiter`
  captures all your changes for the final patch. The delivery layer
  pushes to `amdpilot-org/sglang:amdpilot/sglang-kimi-k26-flydsl-mi355x`
  and `amdpilot-org/aiter:amdpilot/sglang-kimi-k26-flydsl-mi355x`.

## Workspace layout

- SGLang editable install: `/sgl-workspace/sglang` (PR #23381 applied)
- AITER editable install: `/sgl-workspace/aiter` (`dev/kimi-K2.5` branch)
- FlyDSL: `/opt/FlyDSL` (`DSL2_ROOT`)
- MLIR: `/opt/mlir_install` (`MLIR_PATH`)
- Benchmark wrapper: `/workspace/bench_flydsl_k26.sh`
- Accuracy harness: `/workspace/test_harness.py`
- Trial overrides: `/workspace/bench_config.env` (agent writes here)
- Model weights: `/sgl-workspace/models/Kimi-K2.6`
- Server log: `/tmp/sglang_server.log`

## Prior observations (from the blog on MI300X K2.5)

- `fused_moe_kernel_gptq_awq` is **87.8%** of GPU time at concurrency=2
  and **89.7%** at concurrency=40 on the Triton baseline. Anything
  else (AllReduce, GEMM, flash attention) is <3% each.
- FlyDSL's `mixed_moe_gemm_2stage.py` beats Triton by 3-4x on the
  "large" shape `(tokens=16384, model_dim=7168, inter_dim=512, E=384,
  topk=8)` in isolation (8.68 ms vs 12.09 ms for BF16, and 9.77 ms vs
  31.43 ms for W4A16).
- End-to-end on MI300X: **+162.4%** output throughput at concurrency=40,
  **-69.2%** TPOT mean, **-47.0%** TTFT mean.
- GSM8K accuracy is unchanged between baseline and optimized
  (0.96 both, 10-shot 100 samples).
- `--enable-torch-compile` + `--disable-radix-cache` on their own
  contribute meaningfully to the end-to-end number; Phase E and F
  isolate their effect on MI355X K2.6.

## References

- Blog: [Accelerating Kimi-K2.5 on AMD Instinct MI300X: Optimizing Fused MoE with FlyDSL](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)
- Baseline PR: [sgl-project/sglang#23381](https://github.com/sgl-project/sglang/pull/23381) (source branch: [jhinpan/sglang-fork:feat/kimi-k26-mi355x-moe-tuning](https://github.com/jhinpan/sglang-fork/tree/feat/kimi-k26-mi355x-moe-tuning))
- FlyDSL: <https://github.com/ROCm/FlyDSL>
- AITER `dev/kimi-K2.5` branch: <https://github.com/ROCm/aiter/tree/dev/kimi-K2.5>
- Kimi-K2.6 model card: <https://huggingface.co/moonshotai/Kimi-K2.6>
- Upstream Docker tag: `jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420`
  (built from `rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260420`)
