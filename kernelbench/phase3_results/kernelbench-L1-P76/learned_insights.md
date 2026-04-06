# Learned Insights

- **Trial 1**: For 1D conv with large output lengths (174758), MIOpen on MI355X is extremely hard to beat with naive Triton kernels — Triton conv1d was ~100x slower
- **Trial 1**: Triton compilation for large grid sizes (output_length=174758) on AMD can exceed 300s, causing benchmark timeouts. Pre-warming the cache via a separate script is essential
- **Trial 1**: torch.compile on F.conv1d provides no speedup when MIOpen is already the backend
- **Trial 1**: KernelBench score of 60 corresponds to matching PyTorch baseline speed (speedup=1.0) with correct output
- **Trial 1**: For memory-bound conv1d operations on AMD, consider im2col + GEMM approach using rocBLAS/hipBLASLt rather than naive Triton direct convolution
