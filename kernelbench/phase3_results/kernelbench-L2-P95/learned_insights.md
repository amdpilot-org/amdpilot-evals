# Learned Insights

- **Trial 1**: tl.libdevice.tanh is unavailable on ROCm Triton for MI355X; must use manual tanh via tl.math.exp with clamping to [-10, 10]
- **Trial 1**: For KernelBench problem 95 (1024x8192 @ 8192x8192 + activations), baseline PyTorch runs in ~0.999ms; simple activation fusion gives only 1.047x speedup
- **Trial 1**: ROCm Triton requires explicit .to(tl.float32) casts on load and store operations
- **Trial 1**: BLOCK_SIZE should be power-of-2 and multiple of 64 (AMD wavefront size) for MI355X
- **Trial 1**: The elementwise activation fusion alone is not enough for major speedup — the GEMM dominates for large matrix sizes like 8192x8192
- **Trial 2**: For KernelBench problem 95, the GEMM (1024x8192 @ 8192x8192) dominates runtime; activation fusion alone yields only 1.047x speedup
- **Trial 2**: The biggest optimization opportunity is fusing GEMM + activations to eliminate the intermediate global memory write/read of the 1024x8192 output tensor
- **Trial 2**: Score of 60.50 corresponds to 1.047x speedup (0.954ms vs 0.999ms baseline)
- **Trial 3**: Agent got stuck (no output) in trials 2 and 3 of KernelBench problem 95 — likely attempting overly ambitious GEMM fusion that fails silently
- **Trial 3**: For KernelBench, do NOT attempt to write Triton GEMM kernels for large matrices (8192x8192) — use rocBLAS via torch.mm/nn.Linear and fuse only the elementwise activations
- **Trial 3**: When agent produces no output for 2+ consecutive trials, provide extremely concrete step-by-step instructions starting with verification of the existing working code
- **Trial 4**: Agent gets stuck with no output when attempting complex Triton GEMM fusion for large matrices — must explicitly forbid this approach
- **Trial 4**: For 3+ consecutive no-output failures, provide copy-paste-ready code snippets rather than conceptual guidance
- **Trial 4**: torch.compile with mode='max-autotune' on just the activation portion (post-GEMM) is a viable alternative to manual Triton for elementwise fusion
- **Trial 5**: Agent produces no output 4 consecutive trials when instructions are conceptual — must provide complete copy-paste-ready Python files
- **Trial 5**: For KernelBench problem 95, 2D tiled Triton kernel with autotune configs is the correct approach for the activation fusion kernel
- **Trial 5**: tl.sigmoid is available on ROCm Triton and can be used for swish activation
