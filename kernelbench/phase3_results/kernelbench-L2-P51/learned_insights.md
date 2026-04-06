# Learned Insights

- **Trial 1**: For problem 51 (2048x8192 GEMM + post-ops), GEMM dominates at 94.3% of runtime via rocBLAS addmm
- **Trial 1**: ROCm Triton on MI355X: tl.libdevice.tanh is unavailable, must implement GELU manually using tl.math.exp
- **Trial 1**: torch.compile should NOT wrap ModelNew when custom Triton kernels are used — causes zero output or pointer access errors
- **Trial 1**: Fusing only the non-GEMM ops (subtract, mean, logsumexp, gelu, residual) yields only ~1.03x speedup since they represent ~5.7% of runtime
- **Trial 1**: BLOCK_SIZE must be power of 2 and >= dimension size for CDNA4 wavefront alignment on MI355X
- **Trial 1**: LogSumExp on a single-element dimension is identity — can be skipped in the fused kernel
- **Trial 2**: Trial 2 produced no output — agent may have crashed during large code generation or hit a silent error
- **Trial 2**: Score of 60.30 represents the KernelBench composite score (higher=better), not raw latency
- **Trial 2**: For 2048x8192x8192 GEMM, fusing GEMM with post-ops into a single Triton kernel could save one full write+read of the 2048x8192 intermediate tensor (~32MB)
- **Trial 3**: Agent crashed silently in 2 consecutive trials (2 and 3) — likely due to attempting overly complex kernel implementations
- **Trial 3**: For problem 51 with 8192x8192 GEMM, do NOT attempt Triton GEMM fusion — rocBLAS is optimal and Triton GEMM at this scale is error-prone on CDNA4
- **Trial 3**: When agent produces no output, next trial should start with verifying existing code works before making any changes
- **Trial 4**: Agent crashes silently when attempting complex kernel implementations — must enforce step-by-step approach with testing between each change
- **Trial 4**: 3 consecutive no-output failures suggest the agent is either generating too much code or hitting a silent compilation error
- **Trial 4**: For problem 51, the practical optimization ceiling is limited since GEMM is 94.3% of runtime and already uses rocBLAS
- **Trial 5**: Agent has crashed silently in 4 out of 5 trials on problem 51 — must enforce extremely minimal code changes
- **Trial 5**: For problem 51, the practical ceiling is ~1.05x over baseline since GEMM is 94.3% of runtime via rocBLAS
- **Trial 5**: When agent crashes repeatedly, the recovery strategy must prioritize reproducing the existing working score before any modifications
