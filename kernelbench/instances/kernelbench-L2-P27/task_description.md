# KernelBench Level 2 Problem 27: 27_Conv3d_HardSwish_GroupNorm_Mean.py

## Goal

Write an optimized Triton kernel implementation (`ModelNew`) that:
1. Produces the **exact same output** as the PyTorch reference `Model`
2. Is **faster** than the PyTorch baseline
3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)

## PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish activation
    3. GroupNorm  
    4. Mean pooling across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)                             # (B, C, D, H, W)
        x = F.hardswish(x)                           # Nonlinear activation
        x = self.group_norm(x)                       # Normalization over channels
        x = torch.mean(x, dim=[2, 3, 4])             # Mean over spatial dims → (B, C)
        return x

# === Test config ===
batch_size = 1024
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
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
def fused_hardswish_gnorm_mean_kernel(
    x_ptr,           # Input: (B, C, D, H, W) - conv output
    weight_ptr,      # GroupNorm weight: (C,)
    bias_ptr,        # GroupNorm bias: (C,)
    out_ptr,         # Output: (B, C)
    B, C, D, H, W,
    num_groups,
    eps,
    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_outb, stride_outc,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one batch element and a block of channels
    batch_idx = tl.program_id(0)
    channel_block_idx = tl.program_id(1)
    
    channel_start = channel_block_idx * BLOCK_C
    channel_offs = channel_start + tl.arange(0, BLOCK_C)
    channel_mask = channel_offs < C
    
    # Calculate group index for each channel
    channels_per_group = C // num_groups
    group_offs = channel_offs // channels_per_group
    
    # Load weight and bias for these channels
    weight = tl.load(weight_ptr + channel_offs, mask=channel_mask, other=0.0)
    bias = tl.load(bias_ptr + channel_offs, mask=channel_mask, other=0.0)
    
    spatial_size = D * H * W
    group_spatial_size = channels_per_group * D * H * W
    
    # First pass: Compute group mean and variance on HardSwish output
    # We need to accumulate across all channels in the group and all spatial dims
    group_sum = tl.zeros([BLOCK_C], dtype=tl.float32)
    group_sum_sq = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    for d in range(D):
        for h in range(H):
            for w in range(W):
                x_offs = (batch_idx * stride_xb + 
                         channel_offs * stride_xc + 
                         d * stride_xd + 
                         h * stride_xh + 
                         w * stride_xw)
                x = tl.load(x_ptr + x_offs, mask=channel_mask, other=0.0)
                
                # Apply HardSwish: x * relu6(x + 3) / 6
                x_plus_3 = x + 3.0
                x_clipped = tl.maximum(0.0, tl.minimum(6.0, x_plus_3))
                hardswish_x = x * x_clipped / 6.0
                
                group_sum += hardswish_x
                group_sum_sq += hardswish_x * hardswish_x
    
    # Compute per-channel mean (will be refined to group-level)
    channel_mean = group_sum / spatial_size
    channel_var = group_sum_sq / spatial_size - channel_mean * channel_mean
    
    # For proper GroupNorm, we need group-level statistics
    # Aggregate across channels in the same group
    # Use a simple reduction within the block
    group_mean = tl.zeros([BLOCK_C], dtype=tl.float32)
    group_var = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    # Sum across channels in the same group
    for c in range(channels_per_group):
        ch_idx = group_offs * channels_per_group + c
        ch_mask = (ch_idx >= channel_start) & (ch_idx < channel_start + BLOCK_C) & (ch_idx < C)
        # This is simplified - in practice we'd use shared memor
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 27
```
