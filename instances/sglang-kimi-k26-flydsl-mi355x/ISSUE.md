## Summary

Port the [FlyDSL](https://github.com/ROCm/FlyDSL) fused-MoE kernel stack from AMD's [Kimi-K2.5 MI300X optimization blog](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html) to **Kimi-K2.6 on 4x MI355X**, on top of the baseline established by [PR #23381](https://github.com/sgl-project/sglang/pull/23381) (source branch: [`jhinpan/sglang-fork:feat/kimi-k26-mi355x-moe-tuning`](https://github.com/jhinpan/sglang-fork/tree/feat/kimi-k26-mi355x-moe-tuning)).

`fused_moe_kernel_gptq_awq` dominates 88-90% of GPU time in both prefill-heavy and decode-heavy regimes on the current (Triton) path. Replacing it with the FlyDSL mixed-precision (W4A16 + BF16) kernel delivered **+162% output throughput / -69% TPOT / -47% TTFT** on MI300X K2.5 with no GSM8K accuracy regression. We want to measure and capture the corresponding win on MI355X K2.6, on top of PR #23381's already-tuned Triton MoE JSON config.

## Baseline (from PR #23381)

Measured on 4x MI355X (TP=4, `bench_one_batch_server`, BS=1, output=1024):

| input_len | TTFT (s) | Decode (tok/s) | Overall (tok/s) |
| --------- | -------- | -------------- | --------------- |
| 1,024     | 0.44     | 40.24          | 80.47           |
| 2,048     | 0.44     | 40.01          | 120.04          |
| 4,096     | 0.66     | 39.07          | 195.33          |
| 8,192     | 0.70     | 38.05          | 342.44          |
| 16,384    | 1.17     | 35.85          | 609.49          |
| 32,768    | 2.42     | 32.03          | 1056.88         |

Container: `jhinpan/sglang-k26-mi355x:v0.5.10rc0-rocm720-20260420`.

The concurrency=40 `bench_serving` number is **not** published in the PR and will be pinned by Phase 1 of this task (see below).

## Target

Primary metric: `output_throughput_tok_s` at `concurrency=40 random-input=10240 random-output=512`, higher is better.

- **Must**: >= **1.30x Phase 1 baseline** at concurrency=40 (conservative vs the blog's +162% on MI300X).
- **Must**: BS=1 `decode_bs1_in8k` >= **0.98x** PR #23381's published 38.05 tok/s (regression guard).
- **Must**: GSM8K `exact_match,flexible-extract` @ limit=50 >= **0.90** (blog: 0.96 @ limit=100).

## Phased plan

Executed autonomously by the amdpilot Kimi-K2.6 ralph-mode executor. Each phase is a trial bucket.

| Phase | Change                                                             | Rationale                                                           |
| ----- | ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| A     | Verify FlyDSL + AITER `dev/kimi-K2.5` loads on K2.6 gfx950 weights | Cheap go/no-go; fall through to framework-only if it fails.         |
| B     | `AITER_USE_FLYDSL_MOE_STAGE1=1` only                               | Isolate Stage 1 (gate/up) gain; keep Triton on Stage 2.             |
| C     | + `AITER_USE_FLYDSL_MOE_STAGE2=1`                                  | Full blog recipe.                                                   |
| D     | Sweep `FLYDSL_W4A16_HYBRID` ∈ {unset, `w2_bf16`}                   | Mixed precision tradeoff; blog defaults to `w2_bf16`.               |
| E     | `+ --enable-torch-compile`                                         | Framework-level; blog notes this helps decode CPU launch overhead.  |
| F     | `+ --disable-radix-cache`                                          | Random-input benchmark has no shared prefix; frees KV cache memory. |
| G     | `test_harness.py` on best config                                   | GSM8K accuracy gate; rejects any win with >3pp regression.          |

## Environment variables

Blog-canonical set, sourced by the agent from `/workspace/bench_config.env`:

```
export AITER_USE_FLYDSL_MOE=1
export AITER_USE_FLYDSL_MOE_STAGE1=1
export AITER_USE_FLYDSL_MOE_STAGE2=1
export AITER_ENFORCE_DSL=1
export DSL2_ROOT=/opt/FlyDSL
export MLIR_PATH=/opt/mlir_install
export FLYDSL_W4A16_HYBRID=w2_bf16           # or unset for both-W4A16
export AITER_FLYDSL_MOE_COMPARE=0
export AITER_FLYDSL_MOE_COMPARE_STAGE2=0
export AITER_FLYDSL_DEBUG=0
export TRITON_MAX_CACHE_SIZE=2147483648
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3
export EXTRA_SERVER_FLAGS="--enable-torch-compile --disable-radix-cache"
```

## Immutables

- Model: `/sgl-workspace/models/Kimi-K2.6`, TP=4
- `--decode-attention-backend triton --prefill-attention-backend aiter`
- `--mem-fraction-static 0.85 --context-length 128000 --disable-custom-all-reduce`
- Benchmark wrapper and accuracy harness (`/workspace/bench_flydsl_k26.sh`, `/workspace/test_harness.py`)
- PR #23381's tuned Triton MoE JSON (`E=384,N=256,MI355X,int4_w4a16.json`) — remains as a fallback when FlyDSL is disabled; FlyDSL replaces Triton at runtime via env vars, not by deleting files

## AMDPilot eval

The full hand-authored eval instance lives at [`evals/instances/sglang-kimi-k26-flydsl-mi355x/`](https://github.com/amdpilot-org/amdpilot/tree/main/evals/instances/sglang-kimi-k26-flydsl-mi355x) in the amdpilot repo, including:

- `task.yaml` — `type: optimize`, `phase1_baseline: true`, `frontier_model: true`, 4x MI355X TP=4, 6h budget
- `Dockerfile` — layers FlyDSL `main` + AITER `dev/kimi-K2.5` + MLIR on top of the PR #23381 base image
- `bench_flydsl_k26.sh` — canonical PR #23381 launch + concurrency=40 `bench_serving` (primary) + BS=1 `bench_one_batch_server` (guard)
- `test_harness.py` — Phase G GSM8K accuracy gate via `lm_eval`
- `task_description.md` / `metadata.json`

Dispatcher should **reuse the hand-authored artifacts** rather than re-formulating via the LLM; formulation on an issue this specific is a regression risk (pinned AITER branch + MLIR build + env var ordering). Equivalent commands:

```bash
# Preferred: bypass formulator
uv run amdpilot run evals/instances/sglang-kimi-k26-flydsl-mi355x/task.yaml --hours 6 --clean-start

# Issue-driven: formulator should detect the eval folder via this issue link and reuse
uv run amdpilot submit https://github.com/amdpilot-org/sglang/issues/<THIS_ISSUE_NUMBER>
```

## Delivery

Final patch lands on [`amdpilot-org/sglang:amdpilot/sglang-kimi-k26-flydsl-mi355x`](https://github.com/amdpilot-org/sglang/tree/amdpilot/sglang-kimi-k26-flydsl-mi355x) (and on `amdpilot-org/aiter` for any AITER-side changes), matching the `amdpilot/<task-name>` convention already in use on this fork. Upstream PR to `sgl-project/sglang` is a manual follow-up once trials converge and accuracy is verified.

## Risks / open questions

1. **FlyDSL gfx950 support in practice.** README lists MI350/MI355X as verified, blog demonstrates MI300X only. Phase A is the gate.
2. **AITER `dev/kimi-K2.5` vs K2.6 weights.** PR #23381 states "K2.6 reuses the K2.5 MoE/MLA architecture", so the kernels should apply; Phase A confirms.
3. **MLIR install at Docker build time.** Dockerfile tries `rocm-llvm-dev` apt, falls back to FlyDSL's `scripts/build_llvm.sh` (30-60 min one-time).
4. **Concurrency=40 bench runtime.** ~5-7 min per call; Phase 1 (<=90 min) + 10-15 executor trials fit in 6h, but bump `max_total_hours` if Phase 1 stretches.

## References

- Blog: <https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html>
- Baseline PR: <https://github.com/sgl-project/sglang/pull/23381>
- FlyDSL: <https://github.com/ROCm/FlyDSL>
- AITER `dev/kimi-K2.5`: <https://github.com/ROCm/aiter/tree/dev/kimi-K2.5>
- Kimi-K2.6 model card: <https://huggingface.co/moonshotai/Kimi-K2.6>
