# KernelBench Level 2 Problem 17: 17_Conv2d_InstanceNorm_Divide.py

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
    Simple model that performs a convolution, applies Instance Normalization, and divides by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = x / self.divide_by
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
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
def instance_norm_divide_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    eps,
    divide_by,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    channel_id = tl.program_id(0)
    
    # Total elements per channel (across batch and spatial dimensions)
    hw = H * W
    hw_total = N * hw
    
    # Base offset for this channel
    base_offset = channel_id * hw
    
    # Stride to move between samples in batch for this channel
    batch_stride = C * hw
    
    # First pass: compute mean for this channel
    mean_sum = 0.0
    for n in range(N):
        sample_offset = n * batch_stride
        for block_start in range(0, hw, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hw
            idx = sample_offset + base_offset + offsets
            x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
            mean_sum += tl.sum(x_vals, axis=0)
    
    mean = mean_sum / hw_total
    
    # Second pass: compute variance for this channel
    var_sum = 0.0
    for n in range(N):
        sample_offset = n * batch_stride
        for block_start in range(0, hw, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hw
            idx = sample_offset + base_offset + offsets
            x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
            diff = x_vals - mean
            var_sum += tl.sum(diff * diff, axis=0)
    
    variance = var_sum / hw_total
    std = tl.sqrt(variance + eps)
    
    # Third pass: normalize and divide
    for n in range(N):
        sample_offset = n * batch_stride
        for block_start in range(0, hw, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hw
            idx = sample_offset + base_offset + offsets
            x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
            normalized = (x_vals - mean) / std
            out_vals = normalized / divide_by
            tl.store(out_ptr + idx, out_vals, mask=mask)


def triton_instance_norm_divide(x: torch.Tensor, eps: float = 1e-5, divide_by: float = 2.0):
    """
    Fused InstanceNorm2d + division by constant using Triton.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 256
    
    # One program per channel
    grid = (C,)
    
    instance_norm_divide_kernel[grid](
        x, out,
        N, C, H, W,
        eps, divide_by,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies fused Instance Normalization + division.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv =
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 17
```
