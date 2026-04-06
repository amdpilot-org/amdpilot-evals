# Learned Insights

- **Trial 1**: Triton on ROCm does not support `for i in range(runtime_value)` — loop bounds must be constexpr or use tl.static_range with constexpr values
- **Trial 1**: For 3D max pooling with shape (16,32,128,128,128), pointer offsets exceed 2^31 — must use explicit stride-based indexing with proper int64 handling
- **Trial 1**: Block processing with tl.arange (256 elements per program) is essential — one-element-per-program has too much launch overhead
- **Trial 1**: Hardcoding kernel_size=3 and manually unrolling the 27-position window avoids Triton loop construct limitations
- **Trial 1**: 3D max pooling baseline on MI355X: PyTorch nn.MaxPool3d runs at 3.44ms; Triton block kernel achieves 1.94ms (1.77x speedup)
- **Trial 2**: Trial 2 produced no output — agent may have gotten stuck on implementation without ever running the benchmark
- **Trial 2**: Always run the benchmark first to confirm the existing kernel still works before attempting optimizations
- **Trial 3**: Trial 2 and 3 both produced no output — agent is getting stuck on implementation without running benchmarks
- **Trial 3**: Always run the benchmark FIRST before attempting any changes to confirm the environment works
- **Trial 3**: With limited time, focus on incremental optimizations (block size tuning, cache hints) rather than kernel rewrites
- **Trial 4**: 3D max pooling Triton kernel with BLOCK_SIZE=256 and manually unrolled 3x3x3 window achieves 1.77x speedup over PyTorch nn.MaxPool3d on MI355X
- **Trial 4**: Agent repeatedly stalls when asked to optimize an already-working Triton kernel — providing very explicit file paths and 'run benchmark first' instructions across 3 trials was insufficient to unstick it
- **Trial 4**: For KernelBench problem 43 (MaxPool3d), the score of 67.70 corresponds to ~1.94ms kernel time vs 3.44ms PyTorch baseline
- **Trial 4**: When an agent fails 3 consecutive trials with no output, it's better to stop and preserve the existing result than burn remaining time
