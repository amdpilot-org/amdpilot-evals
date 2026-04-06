# KernelBench Level 2 Problem 37: 37_Matmul_Swish_Sum_GroupNorm.py

## Goal

Write an optimized Triton kernel implementation (`ModelNew`) that:
1. Produces the **exact same output** as the PyTorch reference `Model`
2. Is **faster** than the PyTorch baseline
3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)

## PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x

batch_size = 32768
in_features = 1024
out_features = 4096
num_groups = 64
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
```

## AMD ROCm Triton Constraints (CRITICAL)

You are writing Triton kernels for AMD Instinct MI355X (gfx950, CDNA4) with ROCm.

### Known Issues - You MUST follow these rules:

1. **`tl.math.tanh` is UNAVAILABLE** on ROCm Triton. Use manual implementation:
   ```python
   x_clamped = tl.maximum(tl.minimum(x, 10.0), -10.0)
   exp_2x = tl.math.exp(2.0 * x_clamped)
   tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
   ```

2. **`tl.libdevice.*` is UNAVAILABLE** on ROCm. Do NOT use `tl.libdevice.tanh`,
   `tl.libdevice.exp`, etc. Use `tl.math.exp` or manual implementations.

3. **Wavefront size is 64** (not 32 like NVIDIA). BLOCK_SIZE values that are
   multiples of 64 align better with hardware.

4. **Cast output to target dtype explicitly**: compute in float32, cast back on store:
   ```python
   x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
   y = compute(x)
   tl.store(out_ptr, y.to(tl.float32), mask=mask)
   ```

5. **BLOCK_SIZE selection**: Use `triton.next_power_of_2(N)` for the hidden dimension.

6. **Common kernel pattern**:
   ```python
   @triton.jit
   def _kernel(X_ptr, Y_ptr, stride_x, stride_y, N, BLOCK_SIZE: tl.constexpr):
       row = tl.program_id(0)
       cols = tl.arange(0, BLOCK_SIZE)
       mask = cols < N
       x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
       y = compute(x)
       tl.store(Y_ptr + row * stride_y + cols, y.to(tl.float32), mask=mask)
   ```

## Previous Failed Attempt

A previous single-shot attempt failed. The output was incorrect.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_matmul_swish_bias_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for: out = Swish(x @ weight.T + bias)
    Swish(x) = sigmoid(x) * x
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute starting positions
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create ranges for this block
    m_range = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_range = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for bounds checking
    m_mask = m_range < batch_size
    n_mask = n_range < out_features
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform matrix multiplication
    for k in range(0, in_features, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_range < in_features
        
        # Load x block
        x_ptrs = x_ptr + m_range[:, None] * in_features + k_range[None, :]
        x = tl.load(x_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)
        
        # Load weight block (weight is out_features x in_features, need to transpose)
        weight_ptrs = weight_ptr + k_range[:, None] * out_features + n_range[None, :]
        weight = tl.load(weight_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)
        
        # Accumulate
        accumulator += tl.dot(x, weight)
    
    # Add bias
    bias = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0)
    accumulator += bias[None, :]
    
    # Apply Swish activation: sigmoid(x) * x
    sigmoid_val = 1.0 / (1.0 + tl.exp(-accumulator))
    out = accumulator * sigmoid_val
    
    # Store output
    out_ptrs = out_ptr + m_range[:, None] * out_features + n_range[None, :]
    tl.store(out_ptrs, out, mask=(m_mask[:, None] & n_mask[None, :]))


@triton.jit
def group_norm_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    batch_size,
    num_groups,
    features_per_group,
    total_features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for GroupNorm.
    Computes mean and variance per group, then normalizes.
    """
    pid = tl.program_id(0)
    
    # Each program handles one sample in the batch
    batch_idx = pid
    
    # Process each group
    for group_idx in range(num_groups):
        group_start = group_idx * features_per_group
        
        # First pass: compute mean
        mean_acc = 0.0
        for i in range(0, features_per_group, BLOCK_SIZE):
            offsets = group_start + i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < total_features
            x_ptrs = x_ptr + batch_idx * total_features + offsets
            x = tl.load(x_ptrs, m
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 37
```
