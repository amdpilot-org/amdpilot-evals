# Learned Insights

- **Trial 1**: On AMD MI355X, torch.set_float32_matmul_precision('high') enables TF32 and gives ~2.8x speedup for GEMM-heavy workloads
- **Trial 1**: rocBLAS GEMM is ~6x faster than hand-written Triton GEMM kernels for large matrices on MI355X
- **Trial 1**: Triton elementwise kernels (tanh) add ~3% overhead compared to native PyTorch on MI355X
- **Trial 1**: torch.compile with mode='reduce-overhead' fails with manual Triton kernels on ROCm due to CUDA graph capture issues
- **Trial 1**: Split-weight matmuls with TF32 cause numerical divergence (max_diff=0.003) vs concatenated approach due to different accumulation order
- **Trial 1**: For VanillaRNN on MI355X: GEMM i2h=70%, GEMM h2o=25%, elementwise=5% of runtime
- **Trial 1**: KernelBench score 50 = 0.975x speedup; need >1.0x to meaningfully improve score
- **Trial 2**: For VanillaRNN (KernelBench L3 P33) on MI355X, the PyTorch reference with TF32 is extremely hard to beat - rocBLAS GEMM is ~6x faster than hand-written Triton GEMM for the matrix sizes involved (batch=8, input=1024, hidden=256)
- **Trial 2**: When GEMM dominates runtime (95%) and rocBLAS is optimal, Triton-based optimization has very limited headroom - focus should be on fused kernels that eliminate memory round-trips rather than replacing individual GEMM calls
- **Trial 2**: Agent produced no output in trial 2 after receiving complex multi-approach hints - simpler, more focused hints may be more effective
- **Trial 2**: KernelBench score of 50 corresponds to ~0.975x speedup (slightly slower than reference) - achieving score >50 requires actually beating the optimized PyTorch baseline
