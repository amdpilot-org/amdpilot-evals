# KernelBench Level 2 Problem 32: 32_Conv2d_Scaling_Min.py

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
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.conv(x)
        x = x * self.scale_factor
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum along channel dimension
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
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

A previous single-shot attempt failed. Error: `at 21:14:
):
    # Each program handles one (batch, height, width) position
    pid = tl.program_id(0)

    # Decode the position
    batch_idx = pid // (height * width)
    hw_idx = pid % (height ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def scale_and_min_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    scale_factor,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles one (batch, height, width) position
    pid = tl.program_id(0)
    
    # Decode the position
    batch_idx = pid // (height * width)
    hw_idx = pid % (height * width)
    h_idx = hw_idx // width
    w_idx = hw_idx % width
    
    # Initialize min value to infinity
    min_val = tl.float32(float('inf'))
    
    # Iterate over channels in blocks
    for c_start in range(0, channels, BLOCK_SIZE_C):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < channels
        
        # Calculate the index in the flattened tensor
        # Shape: (batch, channels, height, width)
        # Index = batch * channels * height * width + c * height * width + h * width + w
        base_offset = batch_idx * channels * height * width + h_idx * width + w_idx
        offsets = base_offset + c_offsets * height * width
        
        # Load values
        x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
        
        # Apply scale factor
        x = x * scale_factor
        
        # Find minimum in this block
        block_min = tl.min(x, axis=0)
        
        # Update global minimum
        min_val = tl.minimum(min_val, block_min)
    
    # Store the result
    # Output shape: (batch, 1, height, width)
    out_offset = batch_idx * height * width + h_idx * width + w_idx
    tl.store(out_ptr + out_offset, min_val)


def triton_scale_and_min(x: torch.Tensor, scale_factor: float):
    """
    Applies scaling and min reduction along channel dimension using Triton.
    Input: (batch, channels, height, width)
    Output: (batch, 1, height, width)
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    # Prepare output tensor
    out = torch.empty((batch_size, 1, height, width), dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE_C = 32  # Tunable parameter for channel block size
    
    # Number of (batch, height, width) positions
    n_positions = batch_size * height * width
    
    # Determine the number of blocks needed
    grid = (n_positions,)
    
    # Launch the Triton kernel
    scale_and_min_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        scale_factor,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, scales the output, 
    and then applies a minimum operation using Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 32
```
