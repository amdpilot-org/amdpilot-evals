# KernelBench Level 2 Problem 79: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max.py

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
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = x * self.multiplier
        x = torch.max(x, dim=1)[0]
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]
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
import triton
import triton.language as tl


@triton.jit
def compute_instance_norm_stats_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size,
    channels,
    spatial_size,
    eps,
    BLOCK_SPATIAL: tl.constexpr,
):
    """
    Compute mean and reciprocal std for instance normalization.
    Each program handles one (batch, channel) pair.
    """
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    if channel_idx >= channels:
        return
    
    # Compute mean over spatial dimensions
    mean_sum = 0.0
    for block_start in range(0, spatial_size, BLOCK_SPATIAL):
        spatial_offsets = block_start + tl.arange(0, BLOCK_SPATIAL)
        mask = spatial_offsets < spatial_size
        
        linear_indices = ((batch_idx * channels + channel_idx) * spatial_size + spatial_offsets)
        vals = tl.load(x_ptr + linear_indices, mask=mask, other=0.0)
        mean_sum += tl.sum(vals, axis=0)
    
    mean = mean_sum / spatial_size
    
    # Compute variance
    var_sum = 0.0
    for block_start in range(0, spatial_size, BLOCK_SPATIAL):
        spatial_offsets = block_start + tl.arange(0, BLOCK_SPATIAL)
        mask = spatial_offsets < spatial_size
        
        linear_indices = ((batch_idx * channels + channel_idx) * spatial_size + spatial_offsets)
        vals = tl.load(x_ptr + linear_indices, mask=mask, other=0.0)
        diffs = vals - mean
        var_sum += tl.sum(diffs * diffs, axis=0)
    
    var = var_sum / spatial_size
    std = tl.sqrt(var + eps)
    rstd = 1.0 / std
    
    # Store statistics
    stat_idx = batch_idx * channels + channel_idx
    tl.store(mean_ptr + stat_idx, mean)
    tl.store(rstd_ptr + stat_idx, rstd)


@triton.jit
def fused_norm_clamp_mult_max_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    multiplier_ptr,
    batch_size,
    channels,
    spatial_size,
    clamp_min,
    clamp_max,
    BLOCK_CHANNEL: tl.constexpr,
):
    """
    Apply: instance_norm -> clamp -> multiply -> max over channels
    Each program handles one (batch, spatial) pair.
    """
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Initialize max value
    max_val = -1e10
    
    # Process channels in blocks
    for channel_block_start in range(0, channels, BLOCK_CHANNEL):
        channel_offsets = channel_block_start + tl.arange(0, BLOCK_CHANNEL)
        channel_mask = channel_offsets < channels
        
        # Load statistics for this block of channels
        stat_indices = batch_idx * channels + channel_offsets
        means = tl.load(mean_ptr + stat_indices, mask=channel_mask, other=0.0)
        rstds = tl.load(rstd_ptr + stat_indices, mask=channel_mask, other=0.0)
        
        # Load input values for this block
        linear_indices = ((batch_idx * channels + channel_offsets) * spatial_size + spatial_idx)
        vals = tl.load(x_ptr + linear_indices, mask=channel_mask, other=0.0)
        
        # Load multiplier
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 79
```
