# KernelBench Level 2 Problem 75: 75_Gemm_GroupNorm_Min_BiasAdd.py

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
    Model that performs a GEMM, Group Normalization, Minimum operation, and Bias addition.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        x = torch.min(x, dim=1, keepdim=True)[0] 
        x = x + self.bias
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 512
bias_shape = (1, out_features, 1, 1)

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

A previous single-shot attempt failed. Error: `at 18:14:
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Min reduction along channel dimension + Bias addition.
    Input x: (batch_size, num_channels)
    Input bias: (1,) or scala...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def groupnorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    num_channels,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Group Normalization.
    Input: (batch_size, num_channels)
    Output: (batch_size, num_channels)
    """
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    channels_per_group = num_channels // num_groups
    group_start = group_idx * channels_per_group
    
    # Compute mean for this group
    mean = tl.zeros([1], dtype=tl.float32)
    for start in range(0, channels_per_group, BLOCK_SIZE):
        offsets = group_start + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_channels
        x = tl.load(x_ptr + batch_idx * num_channels + offsets, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / channels_per_group
    
    # Compute variance for this group
    var = tl.zeros([1], dtype=tl.float32)
    for start in range(0, channels_per_group, BLOCK_SIZE):
        offsets = group_start + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_channels
        x = tl.load(x_ptr + batch_idx * num_channels + offsets, mask=mask, other=0.0)
        var += tl.sum((x - mean) * (x - mean), axis=0)
    var = var / channels_per_group
    
    # Normalize and apply weight/bias
    for start in range(0, channels_per_group, BLOCK_SIZE):
        offsets = group_start + start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_channels
        x = tl.load(x_ptr + batch_idx * num_channels + offsets, mask=mask, other=0.0)
        x_norm = (x - mean) / tl.sqrt(var + eps)
        
        if weight_ptr is not None:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
            x_norm = x_norm * weight
        
        if bias_ptr is not None:
            gn_bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
            x_norm = x_norm + gn_bias
        
        tl.store(out_ptr + batch_idx * num_channels + offsets, x_norm, mask=mask)


@triton.jit
def min_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Min reduction along channel dimension + Bias addition.
    Input x: (batch_size, num_channels)
    Input bias: (1,) or scalar
    Output: (batch_size, 1)
    """
    batch_idx = tl.program_id(0)
    
    # Find minimum across all channels
    min_val = tl.float32(float('inf'))
    
    for start in range(0, num_channels, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_channels
        x = tl.load(x_ptr + batch_idx * num_channels + offsets, mask=mask, other=float('inf'))
        block_min = tl.min(x, axis=0)
        min_val = tl.minimum(min_val, block_min)
    
    # Add bias
    bias = tl.load(bias_ptr)
    out = min
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 75
```
