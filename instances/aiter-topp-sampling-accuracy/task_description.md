# Top-P Sampling Accuracy Issues in AIter

## Problem

Top-p (nucleus) sampling in the AIter library produces incorrect results under
certain conditions, leading to observable quality regressions in downstream LLM
inference.

## Symptoms

1. **Repetitive and corrupted text generation under tensor parallelism (TP).**
   When running multi-GPU inference with tensor parallelism (e.g., TP=2 or
   higher), top-p sampling produces noticeably degraded output. Generations
   become repetitive, incoherent, or contain corrupted tokens. The issue does
   not manifest (or is much less apparent) with TP=1.

2. **Accuracy regression on reasoning benchmarks.** Benchmark scores such as
   GSM8K drop significantly when using top-p sampling compared to expected
   baselines. The regression is consistent and reproducible across runs,
   indicating a systematic error rather than sampling variance.

3. **Non-deterministic divergence across TP ranks.** Different tensor-parallel
   ranks that should be producing coordinated sampling decisions appear to
   diverge, suggesting that the random state used during sampling is not
   consistent across ranks.

## Scope

The bug affects the top-p sampling kernels and associated Python utilities in
the AIter package.

## Environment

- AMD GPUs (MI300X / MI250X series)
- ROCm platform
- AIter library used as a backend for SGLang or similar LLM serving frameworks

## Task

Identify and fix the root cause(s) of the top-p sampling accuracy issues.
The fix should:

- Ensure correct and reproducible sampling under tensor parallelism
- Restore benchmark accuracy (e.g., GSM8K) to expected levels
- Produce correct results on AMD GPU hardware
