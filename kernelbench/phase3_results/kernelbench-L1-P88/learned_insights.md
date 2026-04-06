# Learned Insights

- **Trial 1**: KernelBench score of 100.0 is the maximum; no further optimization stages needed once achieved.
- **Trial 1**: On ROCm Triton, tl.math.tanh is unavailable; use manual implementation: clamp to [-10,10], compute exp(2x), then (exp(2x)-1)/(exp(2x)+1).
- **Trial 1**: GELU activation fusion into a single Triton kernel eliminates 5+ separate PyTorch elementwise kernels (mul, add, pow, tanh) and achieves ~7x speedup on MI355X.
- **Trial 1**: BLOCK_SIZE=4096 works well for large elementwise kernels (67M elements) on CDNA4 with wavefront size 64.
- **Trial 1**: Removing is_cuda assertions is necessary for AMD ROCm compatibility in Triton kernels.
- **Trial 1**: For Level 1 Problem 88, the baseline PyTorch GELU takes ~0.85ms; a fused Triton kernel achieves ~0.12ms.
