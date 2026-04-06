# Learned Insights

- **Trial 1**: For problem 94 (1024x8192 GEMM + bias+hardtanh+mish+GroupNorm), GEMM via hipBLAS dominates at 90.8% of runtime
- **Trial 1**: Fused Triton kernel for bias+hardtanh+mish achieves 0.18ms vs 0.61ms for separate PyTorch ops on MI355X
- **Trial 1**: Triton GroupNorm (two-pass) was 2x slower than PyTorch native GroupNorm on ROCm - avoid unless fusing with other ops to save memory traffic
- **Trial 1**: ROCm Triton constraints: use tl.math.exp instead of tl.libdevice.tanh, manual tanh via (exp(2x)-1)/(exp(2x)+1), explicit .to(tl.float32) casts, BLOCK_SIZE multiple of 64
- **Trial 1**: With 256 groups and 8192 out_features, each group has 32 channels - small enough for a single Triton program to handle per (batch, group)
- **Trial 2**: Trial 2 produced no output - agent may have gotten stuck trying complex optimizations without a fallback plan
- **Trial 2**: For problem 94, the main fusion opportunity is combining bias+hardtanh+mish+GroupNorm into one kernel to eliminate a full (1024,8192) tensor read+write
- **Trial 2**: With 256 groups and 8192 channels, each group has exactly 32 channels - small enough to compute GroupNorm statistics entirely in registers without a two-pass algorithm
- **Trial 3**: Agent crashed/hung in trials 2 and 3 of problem 94 - needs extremely concrete code templates and strict time limits
- **Trial 3**: For problem 94, the fused bias+hardtanh+mish+GroupNorm kernel should use grid=(1024, 256) with 32 channels per program - all fits in registers
- **Trial 3**: When agent produces no output in consecutive trials, provide near-complete code to avoid implementation rabbit holes
- **Trial 4**: Agent crashes/hangs on trials 2-4 for problem 94 - needs near-complete code to avoid implementation rabbit holes
- **Trial 4**: For problem 94, fusing bias+hardtanh+mish+GroupNorm into one kernel saves one (1024,8192) tensor read+write, which should improve the non-GEMM portion significantly
- **Trial 4**: With 256 groups and 8192 channels, channels_per_group=32 which is a power of 2 and fits perfectly in a single Triton program using tl.arange(0, 32)
- **Trial 5**: Agent has crashed/hung 4 consecutive trials (2-5) on problem 94 - needs complete copy-paste code with zero ambiguity
- **Trial 5**: For problem 94, the input after hardtanh is clamped to [-1,1], so softplus values are bounded and numerically stable without special handling
- **Trial 5**: When agent repeatedly fails, provide explicit fallback instructions to at least reproduce previous best
