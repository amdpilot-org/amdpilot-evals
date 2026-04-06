# Learned Insights

- **Trial 1**: For problem 62 on MI355X: GEMM=86%, GroupNorm=11%, elementwise=3% of runtime at sizes batch=1024, input=hidden=8192, groups=512
- **Trial 1**: rocBLAS GEMM cannot be beaten by hand-written Triton matmul on MI355X for large square matrices
- **Trial 1**: PyTorch GroupNorm is highly optimized - naive Triton replacements with many small blocks (channels_per_group=16) are slower due to launch overhead
- **Trial 1**: torch.compile has import errors with Triton on ROCm ('specialize_impl' missing) - avoid combining them
- **Trial 1**: Fusing LeakyReLU+double into one in-place Triton kernel saves only ~3% (elementwise portion) - need to target GroupNorm fusion for meaningful gains
- **Trial 1**: With 512 groups of 16 channels, the GroupNorm fusion kernel must avoid launching batch×groups blocks. Use one block per batch row processing all groups to minimize launch overhead.
- **Trial 2**: Trial 2 produced no output - agent may need explicit instruction to start from existing working solution rather than starting from scratch
- **Trial 2**: Fusing GroupNorm+LeakyReLU+double into one Triton kernel with grid=(batch_size,) and looping over 512 groups of 16 channels per block is the most promising optimization path for this problem
- **Trial 3**: Agent failed to produce output in trials 2 and 3 - needs extremely concrete instructions with minimal exploration to avoid timing out
- **Trial 3**: For GroupNorm fusion with groups=512, channels_per_group=16: use grid=(batch_size,) with a loop over groups inside each program to avoid excessive block launches
- **Trial 4**: Agent repeatedly fails with no output in trials 2-4 - needs extremely prescriptive copy-paste code, not exploration instructions
- **Trial 4**: For channels_per_group=16, tl.arange(0, 16) fits easily in registers - fused GroupNorm kernel with grid=(batch_size,) looping over groups should work
- **Trial 4**: State dict key mapping between original Model and ModelNew is a common correctness pitfall - gn.weight/gn.bias must map correctly
- **Trial 5**: For KernelBench problem 62 (Matmul+GroupNorm+LeakyReLU+Sum), the operation is GEMM-dominated (86%) — optimization ceiling is very low
- **Trial 5**: With groups=512 and channels_per_group=16, fused GroupNorm kernels struggle: per-group blocks (524K launches) have too much overhead, and per-row blocks with group loops are slower than PyTorch's native GroupNorm
- **Trial 5**: Agent repeatedly produced no output across 4 trials — when this pattern emerges, providing copy-paste code may not help if the agent runs out of time during compilation or exploration before reaching execution
- **Trial 5**: Score of 60.10 achieved with minimal Triton: only fusing LeakyReLU+double into a single in-place kernel while keeping rocBLAS GEMM and PyTorch GroupNorm
- **Trial 5**: torch.compile has import errors with Triton on ROCm ('specialize_impl' missing from triton.runtime.jit) — this path is broken on the test environment
