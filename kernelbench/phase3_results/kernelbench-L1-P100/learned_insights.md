# Learned Insights

- **Trial 1**: For KernelBench hinge loss (32768x32768): row-wise reduction with one block per row and BLOCK_SIZE=32768 achieves 7x speedup over PyTorch
- **Trial 1**: Atomic add approaches for global reduction on AMD ROCm cause GPU memory access faults - use row-wise reduction with final PyTorch sum instead
- **Trial 1**: 1D targets broadcasting across columns: load target_val once per row for efficiency rather than explicit broadcasting
- **Trial 1**: AMD MI355X wavefront size is 64 - BLOCK_SIZE should be multiple of 64 for alignment
- **Trial 1**: Hinge loss is memory-bound - fusing elementwise computation (clamp(1 - pred*target, 0)) with per-row reduction minimizes global memory traffic
- **Trial 1**: Score 100.0 is the maximum in KernelBench scoring - no need for further optimization stages once achieved
