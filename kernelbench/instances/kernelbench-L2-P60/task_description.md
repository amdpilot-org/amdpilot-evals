# KernelBench Level 2 Problem 60: 60_ConvTranspose3d_Swish_GroupNorm_HardSwish.py

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
    Model that performs a 3D transposed convolution, applies Swish activation, 
    group normalization, and then HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = self.group_norm(x)
        x = torch.nn.functional.hardswish(x)  # HardSwish activation
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]
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
def fused_swish_groupnorm_hardswish_kernel(
    x_ptr,           # Input tensor pointer (after conv_transpose)
    out_ptr,         # Output tensor pointer
    gamma_ptr,       # GroupNorm weight
    beta_ptr,        # GroupNorm bias
    N,               # Batch size
    G,               # Number of groups
    C,               # Total channels
    D,               # Depth
    H,               # Height
    W,               # Width
    eps,             # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one group of one sample
    pid = tl.program_id(0)
    n = pid // G  # Batch index
    g = pid % G   # Group index
    
    # Channels per group
    C_g = C // G
    
    # Start channel for this group
    c_start = g * C_g
    
    # Total elements per group per sample
    elements_per_group = C_g * D * H * W
    
    # Compute mean and variance for this group
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = 0
    
    # First pass: compute statistics
    for block_idx in range(0, elements_per_group, BLOCK_SIZE):
        offsets = block_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_group
        
        # Load values (flattened index within group)
        flat_idx = n * C * D * H * W + c_start * D * H * W + offsets
        # Convert flat index to NCDHW coordinates
        # offset = c * D * H * W + d * H * W + h * W + w
        c_off = offsets // (D * H * W)
        rem = offsets % (D * H * W)
        d_off = rem // (H * W)
        rem = rem % (H * W)
        h_off = rem // W
        w_off = rem % W
        
        actual_idx = n * C * D * H * W + (c_start + c_off) * D * H * W + d_off * H * W + h_off * W + w_off
        x_val = tl.load(x_ptr + actual_idx, mask=mask, other=0.0)
        
        # Apply Swish: sigmoid(x) * x
        swish_val = x_val / (1.0 + tl.exp(-x_val)) * x_val
        
        sum_val += tl.where(mask, swish_val, 0.0)
        sum_sq += tl.where(mask, swish_val * swish_val, 0.0)
        count += tl.sum(tl.where(mask, 1, 0))
    
    # Reduce across blocks (simplified - in practice would need atomic ops or multi-pass)
    total_sum = tl.sum(sum_val)
    total_sum_sq = tl.sum(sum_sq)
    
    mean = total_sum / count
    var = total_sum_sq / count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Second pass: normalize and apply activations
    for block_idx in range(0, elements_per_group, BLOCK_SIZE):
        offsets = block_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_group
        
        c_off = offsets // (D * H * W)
        rem = offsets % (D * H * W)
        d_off = rem // (H * W)
        rem = rem % (H * W)
        h_off = rem // W
        w_off = rem % W
        
        actual_idx = n * C * D * H * W + (c_start + c_off) * D * H
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 60
```
