# KernelBench Level 2 Problem 24: 24_Conv3d_Min_Softmax.py

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
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)
        x = torch.min(x, dim=self.dim)[0]  # Apply minimum along the specified dimension
        x = torch.softmax(x, dim=1)  # Apply softmax along the channel dimension
        return x

batch_size = 128
in_channels = 3
out_channels = 24  # Increased output channels
D, H, W = 24, 32, 32  # Increased depth
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
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
def min_softmax_fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    out_stride_b,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one (batch, height, width) position
    pid = tl.program_id(0)
    
    # Decode the position
    hw_total = height * width
    batch_idx = pid // hw_total
    hw_idx = pid % hw_total
    height_idx = hw_idx // width
    width_idx = hw_idx % width
    
    # Channel offsets for this block
    channel_offsets = tl.arange(0, BLOCK_SIZE_C)
    mask_c = channel_offsets < channels
    
    # Initialize min_vals with first depth slice
    base_offset = (batch_idx * stride_b + 
                   channel_offsets * stride_c + 
                   0 * stride_d + 
                   height_idx * stride_h + 
                   width_idx * stride_w)
    min_vals = tl.load(x_ptr + base_offset, mask=mask_c, other=1e10)
    
    # Iterate through remaining depth slices to find minimum
    for d in range(1, depth):
        base_offset = (batch_idx * stride_b + 
                       channel_offsets * stride_c + 
                       d * stride_d + 
                       height_idx * stride_h + 
                       width_idx * stride_w)
        vals = tl.load(x_ptr + base_offset, mask=mask_c, other=1e10)
        min_vals = tl.minimum(min_vals, vals)
    
    # Apply softmax along channel dimension
    # Find max for numerical stability
    max_val = tl.max(min_vals, axis=0)
    
    # Compute exp(x - max)
    exp_vals = tl.exp(min_vals - max_val)
    
    # Sum for normalization
    sum_exp = tl.sum(exp_vals, axis=0)
    
    # Normalize
    softmax_vals = exp_vals / (sum_exp + 1e-10)
    
    # Store result (output has no depth dimension)
    out_offset = (batch_idx * out_stride_b + 
                  channel_offsets * out_stride_c + 
                  height_idx * out_stride_h + 
                  width_idx * out_stride_w)
    tl.store(out_ptr + out_offset, softmax_vals, mask=mask_c)


def triton_min_softmax(x: torch.Tensor, dim: int):
    """
    Fused kernel that performs min reduction along dim followed by softmax along channel dim (dim=1).
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, depth, height, width = x.shape
    
    # Output shape after min reduction (removes the depth dimension)
    out_shape = (batch_size, channels, height, width)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Strides for input (5D: B, C, D, H, W)
    stride_b, stride_c, stride_d, stride_h, stride_w = x.stride()
    
    # Strides for output (4D: B, C, H, W)
    out_stride_b, out_stride_c, out_stride_h, out_stride_w = out.stride()
    
    # Numb
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 24
```
