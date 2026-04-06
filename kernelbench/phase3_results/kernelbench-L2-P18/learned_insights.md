# Learned Insights

- **Trial 1**: For KernelBench problems with sequential reductions, analyze the mathematical chain to identify identity operations — after sum(dim=1, keepdim=True) produces (batch_size, 1), subsequent max/mean/logsumexp along dim=1 are all identity ops and can be eliminated entirely.
- **Trial 1**: Fusing linear + sum avoids materializing the full (batch_size, out_features) intermediate tensor. The fusion uses weight_sum[k] = sum_j W[j,k] and bias_sum = sum_j b[j], reducing the problem to a simple dot product per batch element.
- **Trial 1**: Pre-computing weight_sum and bias_sum on CPU before launching the Triton kernel is cheap and enables massive memory savings (avoids 1024×8192 intermediate).
- **Trial 1**: BLOCK_SIZE = min(triton.next_power_of_2(in_features), 1024) is a good default for reduction kernels on AMD GPUs.
- **Trial 1**: When storing scalar results in Triton, do not use block-type values with scalar pointers — use tl.store with proper scalar extraction.
