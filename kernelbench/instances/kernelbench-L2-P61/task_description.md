# KernelBench Level 2 Problem 61: 61_ConvTranspose3d_ReLU_GroupNorm.py

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
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x

batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 32, 32, 32
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]
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
def groupnorm_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    C,
    D,
    H,
    W,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute mean and rstd for each group in each batch
    """
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    
    channels_per_group = C // num_groups
    group_start = group_id * channels_per_group
    elements_per_group = channels_per_group * D * H * W
    
    # Compute mean and variance for this group
    group_sum = 0.0
    group_sum_sq = 0.0
    
    for elem_id in range(elements_per_group):
        c_offset = elem_id // (D * H * W)
        remaining = elem_id % (D * H * W)
        d = remaining // (H * W)
        remaining = remaining % (H * W)
        h = remaining // W
        w = remaining % W
        
        channel_idx = group_start + c_offset
        idx = batch_id * stride_n + channel_idx * stride_c + d * stride_d + h * stride_h + w * stride_w
        
        val = tl.load(x_ptr + idx)
        group_sum += val
        group_sum_sq += val * val
    
    mean = group_sum / elements_per_group
    variance = group_sum_sq / elements_per_group - mean * mean
    rstd = 1.0 / tl.sqrt(variance + eps)
    
    stat_idx = batch_id * num_groups + group_id
    tl.store(mean_ptr + stat_idx, mean)
    tl.store(rstd_ptr + stat_idx, rstd)


@triton.jit
def fused_relu_groupnorm_apply_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    stride_n,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    C,
    D,
    H,
    W,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply GroupNorm with ReLU fusion
    Each program handles one channel across all spatial locations
    """
    batch_id = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    channels_per_group = C // num_groups
    group_id = channel_idx // channels_per_group
    
    stat_idx = batch_id * num_groups + group_id
    mean = tl.load(mean_ptr + stat_idx)
    rstd = tl.load(rstd_ptr + stat_idx)
    
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    
    for d in range(D):
        for h in range(H):
            for w in range(W):
                idx = batch_id * stride_n + channel_idx * stride_c + d * stride_d + h * stride_h + w * stride_w
                
                val = tl.load(x_ptr + idx)
                
                # Apply ReLU
                val = tl.maximum(val, 0.0)
                
                # Normalize
                normalized = (val - mean) * rstd
                
                # Apply weight and bias
                normalized = normalized * weight_val + bias_val
                
                tl.store(out_ptr + idx, normalized)


def fused_relu_groupnorm(x, weight,
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 61
```
