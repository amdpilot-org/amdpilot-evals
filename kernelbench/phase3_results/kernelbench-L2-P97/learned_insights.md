# Learned Insights

- **Trial 1**: KernelBench L2P97: GEMM (hipBLASLt matmul for 1024x8192 @ 8192x8192) dominates at 94.6% of runtime. Elementwise ops (BN+bias+div+swish) are only ~5.4%.
- **Trial 1**: KernelBench L2P97: Reference model uses TRAINING mode batch norm - must compute batch statistics from input, not use running_mean/running_var. Using running stats causes ~4.5 max diff error.
- **Trial 1**: KernelBench L2P97: Baseline score 60.20 with 0.979ms Triton vs 0.994ms reference (1.015x speedup). Two-kernel approach: compute_batch_stats_kernel + fused_bn_bias_div_swish_kernel.
- **Trial 1**: KernelBench L2P97: Score formula appears to be roughly 100 * (ref_time / triton_time) * some_factor. Higher speedup = higher score.
- **Trial 1**: AMD Triton: Use tl.math.exp not tl.libdevice.exp, use .to(tl.float32) for dtype conversion.
- **Trial 2**: KernelBench L2P97: Trial 2 produced no output - agent may have attempted something that crashed. Always verify the existing working solution first before attempting new optimizations.
- **Trial 2**: KernelBench L2P97: With GEMM at 94.6%, the only way to get significant speedup is to fuse GEMM with post-ops (eliminating intermediate tensor write/read) or use torch.compile for graph-level fusion.
- **Trial 3**: KernelBench L2P97: Agent crashed with no output in trials 2 and 3 - likely attempted too-complex changes (e.g., Triton GEMM) that caused hangs or OOM. Keep changes incremental.
- **Trial 3**: KernelBench L2P97: Two-kernel approach (batch_stats + fused_bn_bias_div_swish) could be merged into single kernel to eliminate kernel launch overhead and intermediate mean/var memory traffic.
- **Trial 4**: KernelBench L2P97: Agent crashed in 3 consecutive trials (2,3,4) - likely due to attempting Triton GEMM or torch.compile which caused hangs on AMD. Must use conservative incremental approach.
- **Trial 4**: KernelBench L2P97: With only ~5% of runtime in elementwise ops, maximum theoretical speedup from Triton kernel fusion is ~1.05x. Focus on eliminating kernel launch overhead and memory traffic between kernels.
- **Trial 5**: KernelBench L2P97: Agent crashed in trials 2-5, likely from attempting Triton GEMM or torch.compile. Must start from known-working solution and make only tiny changes.
- **Trial 5**: KernelBench L2P97: The maximum achievable improvement is ~5% since GEMM dominates at 94.6%. Focus on eliminating kernel launch overhead by merging two elementwise kernels into one.
