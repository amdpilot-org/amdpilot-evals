# Learned Insights

- **Trial 1**: For KernelBench L2P37: Hybrid approach (PyTorch Linear + Triton elementwise + PyTorch GroupNorm) gives 1.15x speedup
- **Trial 1**: Triton naive GEMM is significantly slower than rocBLAS/hipBLASLt on MI300X — always use PyTorch F.linear for GEMM
- **Trial 1**: Triton GroupNorm 3-pass approach had 2x slowdown and numerical precision issues vs PyTorch GroupNorm
- **Trial 1**: Linear layer already includes bias internally — fusing swish(x + linear_bias) + bias is WRONG, must do swish(x) + bias where x already has linear bias applied
- **Trial 1**: Kernel breakdown: GEMM=60%, GroupNorm=25%, elementwise_fused=15% — GroupNorm is the next target for fusion
- **Trial 1**: GroupNorm with num_groups=64 and out_features=4096 means 64 elements per group — small enough for single-warp reduction in Triton
- **Trial 1**: batch_size=32768 means 32768*64=2M groups total — highly parallelizable
- **Trial 2**: Trial 2 produced zero output — agent may need explicit file paths and step-by-step instructions to avoid getting stuck
- **Trial 2**: Fusing swish+bias+GroupNorm is feasible when group_size=64 fits in a single wavefront — key is float32 accumulation for mean/var
- **Trial 3**: Agent got stuck on trials 2 and 3 with no output — needs extremely explicit code templates and step-by-step instructions
- **Trial 3**: For KernelBench L2P37 fused swish+bias+GroupNorm: group_size=64 fits in single wavefront, 32768*64=2M parallel programs, use float32 accumulators for numerical stability
- **Trial 4**: Agent has failed 3 consecutive trials with no output on KernelBench L2P37 — needs complete copy-paste code templates
- **Trial 4**: For fused swish+bias+GroupNorm: GROUP_SIZE=64 (4096/64 groups), grid=(batch_size*num_groups,), all math in float32
- **Trial 5**: Agent has failed 4 consecutive trials with zero output on KernelBench L2P37 — may need to restart or use simpler approach
- **Trial 5**: For fused swish+bias+GroupNorm kernel: group_size=64 (4096/64), use float32 accumulators, single program per (batch, group) pair, grid=(batch_size*num_groups,)=2M programs
- **Trial 5**: When agent produces no output repeatedly, providing complete copy-paste code in hints is the only viable recovery strategy
