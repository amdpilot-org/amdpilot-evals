# KernelBench Level 2 Problem 83: 83_Conv3d_GroupNorm_Min_Clamp_Dropout.py

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
    Model that performs a 3D convolution, applies Group Normalization, minimum, clamp, and dropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.min(x, torch.tensor(min_value, device=x.device))
        x = torch.clamp(x, min=min_value, max=max_value)
        x = self.dropout(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 64, 64
kernel_size = 3
groups = 8
min_value = 0.0
max_value = 1.0
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p]
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

A previous single-shot attempt failed. Error: `at 65:24:
        bias_val = tl.load(bias_ptr + c_idx)

        norm_val = norm_val * weight_val + bias_val

        # Apply clamp (min operation is redundant since clamp includes min)
        norm...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def groupnorm_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    N, C, D, H, W,
    G,
    eps,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute mean and reciprocal std for each group"""
    pid = tl.program_id(0)
    n_idx = pid // G
    g_idx = pid % G
    
    C_per_G = C // G
    c_start = g_idx * C_per_G
    group_size = C_per_G * D * H * W
    
    # Compute mean - first pass
    sum_val = 0.0
    for block_start in range(0, group_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size
        
        # Decode offset to (c_local, d, h, w)
        w_idx = offsets % W
        h_idx = (offsets // W) % H
        d_idx = (offsets // (W * H)) % D
        c_local = offsets // (W * H * D)
        c_idx = c_start + c_local
        
        # Compute flat index into x
        flat_idx = (n_idx * stride_n + c_idx * stride_c + 
                   d_idx * stride_d + h_idx * stride_h + w_idx * stride_w)
        
        x_val = tl.load(x_ptr + flat_idx, mask=mask, other=0.0)
        sum_val += tl.sum(x_val)
    
    mean = sum_val / group_size
    
    # Compute variance - second pass
    var_sum = 0.0
    for block_start in range(0, group_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size
        
        w_idx = offsets % W
        h_idx = (offsets // W) % H
        d_idx = (offsets // (W * H)) % D
        c_local = offsets // (W * H * D)
        c_idx = c_start + c_local
        
        flat_idx = (n_idx * stride_n + c_idx * stride_c + 
                   d_idx * stride_d + h_idx * stride_h + w_idx * stride_w)
        
        x_val = tl.load(x_ptr + flat_idx, mask=mask, other=0.0)
        diff = x_val - mean
        var_sum += tl.sum(diff * diff)
    
    var = var_sum / group_size
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store mean and rstd for this group
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def fused_groupnorm_apply_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    N, C, D, H, W,
    G,
    min_val,
    max_val,
    dropout_p,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply groupnorm, clamp, and dropout in one kernel"""
    pid = tl.program_id(0)
    n_idx = pid // G
    g_idx = pid % G
    
    C_per_G = C // G
    c_start = g_idx * C_per_G
    group_size = C_per_G * D * H * W
    
    # Load precomputed statistics for this group
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    
    # Process all elements in this group
    for block_start in range(0, group_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size
    
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 83
```
