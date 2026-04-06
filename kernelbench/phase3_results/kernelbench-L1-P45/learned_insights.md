# Learned Insights

- **Trial 1**: ROCm Triton has grid size limitations — 1D grids with >10M elements fail silently; use vectorized processing with BLOCK_SIZE to reduce grid size
- **Trial 1**: Must use PyTorch tensor strides (int64) and cast indices to tl.int64 to avoid overflow on large tensors (e.g., 2048x2048)
- **Trial 1**: torch.cuda.synchronize() after kernel launch is critical for catching errors on ROCm
- **Trial 1**: For AvgPool2D on MI355X: vectorized 1D grid with BLOCK_SIZE=256 processing multiple output elements per program achieves 1.85x speedup over PyTorch nn.AvgPool2d (4.12ms vs 7.64ms for batch=16, channels=64, 2048x2048, kernel_size=11)
- **Trial 1**: One-program-per-output-element approach causes 24x slowdown due to launch overhead; amortizing via vectorization is essential
- **Trial 1**: 2D grid (batch*channels, hw) causes GPU memory access faults/crashes on ROCm Triton
