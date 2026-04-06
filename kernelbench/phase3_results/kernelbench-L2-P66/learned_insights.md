# Learned Insights

- **Trial 2**: KernelBench problem 66: GEMM is 95.7% of runtime (504us), softmax is 1.8% (9us), dropout is negligible
- **Trial 2**: Problem dimensions: batch=128, in_features=16384, out_features=16384, dropout_p=0.2
- **Trial 2**: Test harness eval.py has RNG state issue: lines 777-782 run reference then custom forward without resetting seed, causing dropout mask mismatch
- **Trial 2**: rocBLAS GEMM for this problem size is already near-optimal - don't write custom GEMM kernel
- **Trial 2**: Best score so far is 25.00 - need to at minimum reproduce this
- **Trial 2**: Trial with no output likely means compilation error in Triton kernel - keep kernels simple and test early
- **Trial 3**: Two consecutive trials with no output suggest the Triton kernel has compilation errors - always test with simplest possible kernel first
- **Trial 3**: For KernelBench, a pure PyTorch implementation that matches the reference exactly scores 25.00
- **Trial 3**: With limited time, start from a working PyTorch baseline and incrementally add Triton optimizations
- **Trial 4**: Three consecutive trials with no output means the Triton kernels have compilation errors - must start from working PyTorch and add Triton incrementally
- **Trial 4**: For KernelBench problem 66, pure PyTorch implementation scores 25.00 - this is the safe fallback
- **Trial 4**: With GEMM at 95% of runtime and rocBLAS being optimal, max possible speedup from fusing dropout+softmax is ~5%
- **Trial 5**: KernelBench problem 66: 4 consecutive trials failed with no output when attempting custom Triton kernels - compilation errors are the primary failure mode
- **Trial 5**: Safe fallback for this problem is pure PyTorch matching the reference exactly, which scores 25.00
- **Trial 5**: With GEMM at 95% of runtime on rocBLAS, torch.compile for fusion of dropout+softmax is the only viable optimization path
