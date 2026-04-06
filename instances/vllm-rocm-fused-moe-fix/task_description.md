# Task: ROCm AITER fused MoE correctness

## Symptom

On ROCm builds that use AITER fused mixture-of-experts (MoE) paths, MoE-related code misbehaves: operations can crash during tracing, compilation, or dispatch, or silently diverge from the real kernel behavior. The failure mode is tied to **mismatched interfaces** between the Python “fake” / meta implementation and the real custom operator, and to **incomplete propagation of router scaling** into the ROCm AITER top-k / grouping path.

Specifically:

1. The ROCm AITER fused MoE **fake** (mock / meta) handler does not present the same callable interface as the **real** fused MoE implementation (parameter list diverges), which breaks consumers that rely on consistent signatures across fake and impl.

2. The **`routed_scaling_factor`** used by the router is not consistently accepted and forwarded through the ROCm AITER grouped top-k call chain into the underlying ops, so routed expert weights are not scaled as intended.

Fix the codebase so that fused MoE fake and real implementations agree on their signatures, and so that `routed_scaling_factor` is properly threaded through the relevant ROCm AITER MoE paths. Keep changes minimal and aligned with upstream vLLM style.
