# KernelBench Level 2 Problem 85: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py

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
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
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

A previous single-shot attempt failed. Error: `at 30:22:
    # Decode position: (b, group_id, h, w)
    w = pid % width
    pid = pid // width
    h = pid % height
    pid = pid // height
    group_id = pid % num_groups
    b = pid // num_group...`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def groupnorm_scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    batch_size,
    num_channels,
    height,
    width,
    num_groups,
    channels_per_group,
    eps,
):
    """
    Fused GroupNorm + Scale kernel.
    Each program handles one (batch, group, h, w) position.
    """
    pid = tl.program_id(0)
    
    # Decode position: (b, group_id, h, w)
    w = pid % width
    pid = pid // width
    h = pid % height
    pid = pid // height
    group_id = pid % num_groups
    b = pid // num_groups
    
    channel_start = group_id * channels_per_group
    
    # Load all channels for this group at this spatial position
    channel_offsets = tl.arange(0, channels_per_group)
    channel_indices = channel_start + channel_offsets
    
    # Compute flat indices for all channels
    # Shape: (channels_per_group,)
    base_idx = ((b * num_channels) * height + h) * width + w
    indices = base_idx + channel_indices * (height * width)
    
    # Load values
    x_vals = tl.load(x_ptr + indices)
    
    # Compute mean
    mean = tl.sum(x_vals) / channels_per_group
    
    # Compute variance
    var = tl.sum((x_vals - mean) * (x_vals - mean)) / channels_per_group
    std = tl.sqrt(var + eps)
    
    # Normalize and scale
    normalized = (x_vals - mean) / std
    scale_vals = tl.load(scale_ptr + channel_indices)
    out_vals = normalized * scale_vals
    
    # Store output
    tl.store(out_ptr + indices, out_vals)


@triton.jit
def maxpool_clamp_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    num_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    pool_kernel: tl.constexpr,
    clamp_min,
    clamp_max,
):
    """
    Fused MaxPool + Clamp kernel.
    Each program handles one (batch, channel, out_h, out_w) position.
    """
    pid = tl.program_id(0)
    
    # Decode position: (b, c, out_h, out_w)
    out_w = pid % out_width
    pid = pid // out_width
    out_h = pid % out_height
    pid = pid // out_height
    c = pid % num_channels
    b = pid // num_channels
    
    # Calculate input window start
    in_h_start = out_h * pool_kernel
    in_w_start = out_w * pool_kernel
    
    # Find max in the pooling window
    max_val = -1e10
    
    for ph in range(pool_kernel):
        for pw in range(pool_kernel):
            in_h = in_h_start + ph
            in_w = in_w_start + pw
            
            # Bounds check
            if (in_h < in_height) & (in_w < in_width):
                idx = ((b * num_channels + c) * in_height + in_h) * in_width + in_w
                val = tl.load(x_ptr + idx)
                max_val = tl.maximum(max_val, val)
    
    # Clamp
    clamped = tl.maximum(clamp_min, tl.minimum(max_val, clamp_max))
    
    # Store output
    out_idx = ((b * num_channels + c) * out_height + out_h) * out_width + out_w
    tl.store(out_ptr + out_idx, clamped)


def triton_groupnorm_scale(x, scale, num_groups, eps=1e-5
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 85
```
