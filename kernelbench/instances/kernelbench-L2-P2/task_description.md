# KernelBench Level 2 Problem 2: 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py

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
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x * self.scaling_factor
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x / self.scaling_factor
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]
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
def fused_post_conv_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    hw_size,
    clamp_max,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that:
    1. Adds bias (with proper broadcasting for NCHW format)
    2. Applies the simplified clamp operation (equivalent to the original sequence)
    
    Mathematical simplification:
    Original: clamp(x, 0, 1) * scale -> clamp(., 0, 1) / scale
    With scale=2.0, this is equivalent to: clamp(x, 0, 0.5)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index for each element
    # For NCHW format: channel_idx = (offset // hw_size) % n_channels
    channel_indices = (offsets // hw_size) % n_channels
    
    # Load bias values (bias has shape (n_channels, 1, 1))
    bias = tl.load(bias_ptr + channel_indices, mask=mask, other=0.0)
    
    # Add bias
    x = x + bias
    
    # Apply simplified clamp: equivalent to the original sequence of operations
    # Original: clamp(x, 0, 1) * scale -> clamp(., 0, 1) / scale
    # Simplified: clamp(x, 0, clamp_max) where clamp_max = 0.5 for scale=2.0
    x = tl.maximum(x, 0.0)
    x = tl.minimum(x, clamp_max)
    
    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_post_conv(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    """
    Wrapper function for the fused post-convolution kernel.
    Fuses: bias addition + clamp + scale + clamp + divide
    Into: bias addition + single clamp
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    n_channels = x.shape[1]
    hw_size = x.shape[2] * x.shape[3]  # H * W
    
    # Mathematical simplification: the original sequence is equivalent to clamp(x, 0, 0.5)
    # when scaling_factor = 2.0
    clamp_max = 0.5 * scaling_factor  # General formula: 1.0 / scaling_factor * scaling_factor clipped
    
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_post_conv_kernel[grid](
        x, bias, out, n_elements, n_channels, hw_size, clamp_max,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution followed by 
    fused bias addition and clamping operations.
    
    Optimization: The sequence of operations (clamp, scale, clamp, divide) 
    is mathematically equivalent to a single clamp operation, which we 
    fuse with bias addition in a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, s
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 2
```
