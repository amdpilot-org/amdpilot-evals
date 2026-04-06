# Learned Insights

- **Trial 1**: For 99_Matmul_GELU_Softmax (1024x8192x8192): GEMM is 98% of runtime, GELU+Softmax is 2%
- **Trial 1**: tl.dot on ROCm causes 'unrealized_conversion_cast' LLVM translation errors — avoid writing Triton matmul kernels for large shapes, use rocBLAS via torch.mm/F.linear instead
- **Trial 1**: tl.math.tanh is UNAVAILABLE on ROCm — must use manual: clamp x, exp_2x = tl.math.exp(2*x_clamped), tanh = (exp_2x-1)/(exp_2x+1)
- **Trial 1**: For row-wise Triton kernels over N=8192, BLOCK_SIZE=8192 works (next_power_of_2)
- **Trial 1**: Fusing GELU+Softmax into one Triton kernel saves ~0.1ms but GEMM dominates
- **Trial 1**: Baseline score 60.10 corresponds to 0.946ms new vs 0.951ms reference (1.005x speedup)
- **Trial 2**: Trial 2 produced no agent output — possible timeout or crash before any work was done
- **Trial 2**: For 99_Matmul_GELU_Softmax: with 98% GEMM dominance and tl.dot broken on ROCm, torch.compile(mode='max-autotune') is the most promising optimization lever
- **Trial 2**: Score of 60.10 corresponds to only 1.006x speedup — need torch.compile or dtype tricks to improve GEMM performance
- **Trial 3**: Agent has failed to produce output in trials 2 and 3 — may need extremely explicit step-by-step instructions with complete code
- **Trial 3**: For 99_Matmul_GELU_Softmax: since GEMM is 98% and tl.dot is broken on ROCm, torch.compile(mode='max-autotune') on the linear layer is the main optimization lever
- **Trial 4**: Agent has timed out in 3 consecutive trials — needs extremely concise, copy-paste-ready instructions
- **Trial 4**: For 99_Matmul_GELU_Softmax: torch.compile(mode='max-autotune') on the linear layer is the main lever since GEMM is 98% of runtime
- **Trial 4**: tl.dot on ROCm causes LLVM translation errors for large shapes — must use rocBLAS via torch.mm/F.linear
- **Trial 5**: Agent has failed to produce output in trials 2-5 — may be overwhelmed by complex instructions or hitting environment issues silently
- **Trial 5**: For 99_Matmul_GELU_Softmax: torch.compile(mode='max-autotune') on F.linear is the main optimization lever since GEMM is 98% of runtime and tl.dot is broken on ROCm
