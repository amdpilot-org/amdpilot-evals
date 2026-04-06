# Learned Insights

- **Trial 1**: SqueezeNet Fire Module: ReLU (42.2%) + concat (25.4%) dominate at 67.6% of execution time, conv kernels (MIOpen) are 30.2%
- **Trial 1**: Fused ReLU+concat Triton kernel with BLOCK_SIZE=512 achieves 1.24x speedup (6.05ms vs 7.50ms baseline) on MI355X
- **Trial 1**: BLOCK_SIZE=512 (8 warps of 64) is significantly better than smaller block sizes for this workload on MI355X
- **Trial 1**: Triton kernel that processes output as flat 1D array with NCHW index computation and channel-based branching works well for concat fusion
- **Trial 1**: 1x1 convolutions in SqueezeNet are effectively batched GEMM operations — potential Triton matmul target
- **Trial 2**: Trial 2 produced no output - agent may have gotten stuck on implementation without running benchmark
- **Trial 2**: Working baseline from Trial 1: fused ReLU+concat Triton kernel scoring 62.40 (6.05ms vs 7.50ms)
- **Trial 2**: Next optimization target: 1x1 convolutions (squeeze + expand1x1) which are 30.2% of time as MIOpen calls, replaceable with Triton matmul
- **Trial 3**: Agent has crashed/stalled 2 consecutive trials trying to implement complex 1x1 conv replacement - need incremental approach with early benchmark runs
- **Trial 3**: Working baseline from Trial 1 must be preserved and verified before attempting new optimizations
- **Trial 4**: Agent has failed 3 consecutive trials attempting to replace MIOpen 1x1 convolutions with Triton matmul — this approach is too complex and should be abandoned
- **Trial 4**: When agent produces no output for multiple trials, provide extremely prescriptive step-by-step instructions with early benchmark verification
- **Trial 4**: channels_last memory format can significantly improve conv performance on MI355X and is a low-risk optimization
- **Trial 5**: Agent has crashed/stalled 4 consecutive trials (2-5) attempting complex Triton matmul for 1x1 convolutions — this approach must be abandoned entirely
- **Trial 5**: When agent repeatedly crashes, provide actual code snippets and enforce mandatory benchmark runs between each change
- **Trial 5**: channels_last memory format and torch.compile are low-risk optimizations that should be tried before any manual kernel work on convolutions
