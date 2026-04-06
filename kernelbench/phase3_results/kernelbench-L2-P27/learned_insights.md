# Learned Insights

- **Trial 1**: KernelBench L2P27: Conv3d (MIOpen) is 75% of runtime and cannot be improved with Triton
- **Trial 1**: KernelBench L2P27: HardSwish + GroupNorm + Mean fusion achieves ~3.4% speedup with 3 Triton kernels
- **Trial 1**: KernelBench L2P27: GroupNorm requires group-level statistics (mean/var) that create a reduction dependency, making single-kernel fusion difficult
- **Trial 1**: KernelBench L2P27: BLOCK_SIZE=1024 outperforms BLOCK_SIZE=2048 for grid-stride loops over 16*32*32=16384 spatial elements
- **Trial 1**: KernelBench L2P27: Spatial volume is 16*32*32=16384 elements per channel, total 268M elements across batch*channels*spatial
- **Trial 2**: KernelBench L2P27: Trial 2 agent produced no output — likely got stuck on implementation. Need to ensure agent starts from existing working code.
- **Trial 2**: KernelBench L2P27: batched_transpose at 5.1% (0.41ms) is a viable optimization target — may be eliminated with channels_last_3d memory format
- **Trial 2**: KernelBench L2P27: torch.compile(mode='max-autotune') should be tried as an alternative to manual Triton kernels for the post-conv operations
- **Trial 3**: KernelBench L2P27: Agent got stuck 2 consecutive trials — needs extremely prescriptive step-by-step instructions
- **Trial 3**: KernelBench L2P27: channels_last_3d memory format + torch.compile is an untried approach that could eliminate batched_transpose overhead
- **Trial 3**: KernelBench L2P27: When agent gets stuck, provide fallback approaches (A/B/C) with explicit instructions to move on quickly if one fails
- **Trial 4**: KernelBench L2P27: Agent has failed 3 consecutive trials (2,3,4) with no output — environment or implementation complexity may be blocking
- **Trial 4**: KernelBench L2P27: Must provide fallback to simply re-run existing working code if optimizations fail, to avoid zero-score trials
- **Trial 5**: KernelBench L2P27: Agent has failed 4 consecutive trials (2-5) — ultra-minimal step-by-step instructions needed with emphasis on producing ANY score
- **Trial 5**: KernelBench L2P27: Always verify existing working code first before attempting new optimizations to avoid regression to zero-score
