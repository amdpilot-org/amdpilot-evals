# KernelBench Level 2 Problem 5: 5_ConvTranspose2d_Subtract_Tanh.py

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
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x - self.bias
        x = torch.tanh(x)
        return x

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
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

A previous single-shot attempt failed. Error: `at 35:10:
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Calculate channel index for each element
    # Layout: (batch, channel, height, width) in row-major order
    stride_hw = hei...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def bias_sub_tanh_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for bias subtraction and tanh activation.
    Each program processes BLOCK_SIZE elements.
    Bias is broadcasted across spatial dimensions (bias_shape: channels, 1, 1).
    """
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel index for each element
    # Layout: (batch, channel, height, width) in row-major order
    stride_hw = height * width
    channel_indices = (offsets // stride_hw) % channels
    
    # Load bias values (one per channel)
    bias = tl.load(bias_ptr + channel_indices, mask=mask, other=0.0)
    
    # Compute bias subtraction and tanh activation
    out = tl.math.tanh(x - bias)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bias_sub_tanh(x: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper function for the fused bias subtraction and tanh kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    
    batch_size, channels, height, width = x.shape
    total_elements = x.numel()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Tunable block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    bias_sub_tanh_kernel[grid](
        x,
        bias,
        out,
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, 
    then fused bias subtraction and tanh activation using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # Use fused Triton kernel for bias subtraction and tanh
        x = fused_bias_sub_tanh(x, self.bias)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 5
```
