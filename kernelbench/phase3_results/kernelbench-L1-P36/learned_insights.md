# Learned Insights

- **Trial 1**: AMD Triton tl.program_id(0) returns int32 by default — for grids >33M elements with stride multiplications, cast to int64 with .to(tl.int64) to avoid silent overflow
- **Trial 1**: RMSNorm on (112,64,512,512) has 29.4M independent 64-element reductions — each fits in registers, so 1 program per reduction is optimal
- **Trial 1**: PyTorch achieves 7.02ms on this RMSNorm — theoretical bandwidth limit is ~4.7ms on MI355X
- **Trial 1**: torch.compile achieved 4.2ms (1.66x speedup) but task requires @triton.jit kernels
- **Trial 1**: AMD HIP grid x-dimension limit is 2^31-1, not a practical constraint for 29M programs
