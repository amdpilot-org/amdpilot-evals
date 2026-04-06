# KernelBench Level 2 Problem 15: 15_ConvTranspose3d_BatchNorm_Subtract.py

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
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)  # Subtract mean along spatial dimensions
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
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
def fused_batchnorm_spatial_mean_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that applies BatchNorm and subtracts spatial mean.
    Each program handles one (batch, channel) pair.
    """
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Load BatchNorm parameters for this channel
    weight = tl.load(weight_ptr + channel_id).to(tl.float32)
    bias = tl.load(bias_ptr + channel_id).to(tl.float32)
    running_mean = tl.load(running_mean_ptr + channel_id).to(tl.float32)
    running_var = tl.load(running_var_ptr + channel_id).to(tl.float32)
    
    # Compute spatial dimensions
    spatial_size = depth * height * width
    
    # Compute spatial mean for this (batch, channel)
    spatial_sum = 0.0
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                offset = ((batch_id * channels + channel_id) * spatial_size +
                         (d * height * width + h * width + w))
                val = tl.load(x_ptr + offset).to(tl.float32)
                spatial_sum += val
    
    spatial_mean = spatial_sum / spatial_size
    
    # Apply BatchNorm and subtract spatial mean
    std = tl.sqrt(running_var + eps)
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                offset = ((batch_id * channels + channel_id) * spatial_size +
                         (d * height * width + h * width + w))
                val = tl.load(x_ptr + offset).to(tl.float32)
                # Apply BatchNorm
                normalized = (val - running_mean) / std * weight + bias
                # Subtract spatial mean
                result = normalized - spatial_mean
                tl.store(out_ptr + offset, result)


@triton.jit
def fused_batchnorm_spatial_mean_kernel_optimized(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel using block-based processing for better performance.
    Each program handles a block of spatial elements for one (batch, channel) pair.
    """
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    block_id = tl.program_id(2)
    
    # Load BatchNorm parameters for this channel
    weight = tl.load(weight_ptr + channel_id).to(tl.float32)
    bias = tl.load(bias_ptr + channel_id).to(tl.float32)
    running_mean = tl.load(running_mean_ptr + channel_id).to(tl.float32)
    running_var = tl.load(running_var_ptr + channel_id).to(tl.float32)
    
    # Compute spatial dimensions
    spatial_size = depth * height * wi
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 15
```
