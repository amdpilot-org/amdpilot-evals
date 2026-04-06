# Learned Insights

- **Trial 1**: tl.math.erf is available on ROCm Triton and works for exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
- **Trial 1**: tl.libdevice.tanh and tl.math.tanh are UNAVAILABLE on ROCm Triton — do not use tanh-based GELU approximation
- **Trial 1**: GELU tanh approximation fails correctness check (max diff 0.00025 > 1e-4 tolerance) — must use exact GELU with erf
- **Trial 1**: For conv2d+mul+leakyrelu+gelu pipeline: MIOpen conv dominates at 79.6%, bias add is separate at 10.1%, fused elementwise at 10.2%
- **Trial 1**: Conv bias add runs as a separate elementwise_kernel in MIOpen — can be fused into the post-conv Triton kernel for free speedup
- **Trial 1**: BLOCK_SIZE should be multiple of 64 for CDNA4 wavefront alignment on MI355X
- **Trial 2**: Trial 2 produced zero output — agent may have silently crashed or never started execution
- **Trial 2**: For conv2d+mul+leakyrelu+gelu pipeline: fusing bias add into Triton kernel eliminates 10.1% overhead by passing bias=None to Conv2d and adding bias manually in the fused kernel
- **Trial 2**: Score of 62.00 achieved with fused mul+LeakyReLU+GELU kernel using erf-based exact GELU
- **Trial 3**: Agent has silently failed two consecutive trials (2 and 3) on stage2 — may be getting stuck on complex weight-loading changes when trying to fuse conv bias
- **Trial 3**: For conv bias fusion: safer approach is to keep conv with bias and optimize the fused kernel's launch parameters (BLOCK_SIZE, num_warps) rather than restructuring the model
- **Trial 4**: Agent has silently crashed on trials 2, 3, and 4 — complex model restructuring (like fusing conv bias) causes silent failures
- **Trial 4**: When agent fails silently for multiple trials, give extremely minimal step-by-step instructions with explicit file reads first
- **Trial 4**: Score of 62.00 is achievable with fused mul+LeakyReLU+GELU kernel; bias fusion could push to ~69 (eliminating 10.1% bias overhead)
- **Trial 5**: Agent has silently failed on 4 consecutive trials (2-5) — complex changes cause crashes, need copy-paste-level explicit instructions
- **Trial 5**: For KernelBench problem 54: fusing conv bias into Triton kernel should eliminate the separate elementwise_kernel that accounts for 10.1% of runtime
- **Trial 5**: Channel index for NCHW tensor flattened to 1D: channel_idx = (offset // (H*W)) % C
