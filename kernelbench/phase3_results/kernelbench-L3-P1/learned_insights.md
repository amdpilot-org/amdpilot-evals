# Learned Insights

- **Trial 1**: KernelBench score of 50 means matching reference speed (1.0x). Higher scores require being faster than the reference.
- **Trial 1**: For MLP on MI355X: initial baseline 1.19ms reduced to 0.64ms via env vars (GPU_MAX_HW_QUEUES=2, HIP_FORCE_DEV_KERNARG=1, HSA_NO_SCRATCH_RECLAIM=1, PYTORCH_TUNABLEOP_ENABLED=1, TORCH_BLAS_PREFER_HIPBLASLT=1) and torch.set_float32_matmul_precision('high')
- **Trial 1**: torch.compile mode=default works well, mode=max-autotune has high overhead during benchmarking (4.49ms vs 0.64ms)
- **Trial 1**: Naive Triton matmul kernels (5.36ms) cannot compete with rocBLAS for large GEMMs — but fused kernels (GEMM+bias+ReLU) could win by reducing memory traffic
- **Trial 1**: MLP problem has 3 layers: 16384→16384 (GEMM+ReLU), 16384→16384 (GEMM+ReLU), 16384→8192 (GEMM only). GEMM is 85% of time, elementwise ReLU is 5%
- **Trial 1**: Larger Triton block sizes (128x128) were slower than 64x64 for this problem shape (batch=128)
- **Trial 2**: Trial 2 produced no output — agent may have stalled analyzing without acting. Need explicit step-by-step instructions.
- **Trial 2**: BF16 autocast is the most promising untried approach for MLP on MI355X — BF16 GEMMs are ~2x faster than FP32
- **Trial 2**: Score formula: 50 = 1.0x speedup. Need to actually be faster than reference to score above 50.
- **Trial 2**: Fused GEMM+bias+ReLU Triton epilogue could save memory bandwidth but only after BF16 is tried first
- **Trial 3**: Agent stalled on trials 2 and 3 with no output — needs explicit code and step-by-step shell commands
- **Trial 3**: BF16 weight pre-conversion in __init__ avoids per-forward cast overhead
- **Trial 3**: For KernelBench MLP: BF16 GEMMs and torch.compile are the two main levers to try
- **Trial 4**: Agent has stalled 3 consecutive trials (2,3,4) with no output on this task — needs maximally explicit copy-paste commands
- **Trial 4**: BF16 weight pre-conversion + BF16 forward pass is still untried and is the most promising optimization for MLP GEMMs on MI355X
- **Trial 5**: Agent has stalled 4 consecutive trials (2-5) with no output on KernelBench MLP task — needs single copy-paste commands
- **Trial 5**: BF16 weight pre-conversion + BF16 forward with FP32 output cast is still the primary untried optimization for large GEMM MLP on MI355X
