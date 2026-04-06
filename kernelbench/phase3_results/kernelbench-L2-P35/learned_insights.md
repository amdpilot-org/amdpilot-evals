# Learned Insights

- **Trial 1**: ROCm Triton on MI355X: tl.tanh and tl.libdevice.* are unavailable. Use tl.math.exp/tl.math.log and manual tanh: (exp(2x)-1)/(exp(2x)+1) with input clamping to [-10,10]
- **Trial 1**: Grid size must use triton.cdiv(n_elements, BLOCK_SIZE) not n_elements directly - MI355X has hardware grid dimension limits
- **Trial 1**: Triton on ROCm does not support break/continue in loops
- **Trial 1**: BLOCK_SIZE=512 with flat 1D processing gives ~1.33x speedup for fused subtract+hardswish+maxpool+mish kernel
- **Trial 1**: For KernelBench scoring: score = 100 * ref_time / (ref_time + opt_time), so 63.4 corresponds to 1.33x speedup
- **Trial 2**: Trial 2 produced no output - agent may have spent entire time on compilation/debugging without running benchmark. Always run benchmark early to validate.
- **Trial 2**: For fused maxpool kernels, 2D-tiled approach (one thread per output element) has better spatial locality than 1D flat processing with gather
- **Trial 3**: Agent has failed to produce output in 2 consecutive trials on stage2 - likely getting stuck in complex rewrites or compilation loops
- **Trial 3**: With limited time, always run the benchmark first to confirm baseline still works before attempting optimizations
- **Trial 3**: 2D-tiled approach (one thread per output pool element) should improve spatial locality for the fused maxpool kernel
- **Trial 4**: For KernelBench L2 P35 on MI355X: fused subtract+hardswish+maxpool2d+mish Triton kernel with BLOCK_SIZE=512 and 1D flat processing achieves ~1.33x speedup (score 63.40)
- **Trial 4**: Agent repeatedly failed when attempting complex 2D-tiled rewrites of the fused kernel - the 1D flat approach was simpler and worked reliably
- **Trial 4**: When agents fail 3+ consecutive trials with no output, it indicates they're stuck in compile/debug loops - extremely prescriptive hints or stopping is the right call
- **Trial 4**: ROCm Triton constraints summary: no tl.tanh, no tl.libdevice.*, no break/continue in loops, hardware grid dimension limits require triton.cdiv
