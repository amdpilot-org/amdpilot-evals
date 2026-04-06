# KernelBench Level 2 Problem 74: 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max.py

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
    Model that performs a 3D transposed convolution, applies LeakyReLU, multiplies by a learnable parameter, 
    applies LeakyReLU again, and performs a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.leaky_relu(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
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

A previous single-shot attempt failed. Error: `at 25:12:
    stride_w: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.const...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_leaky_mul_leaky_kernel(
    x_ptr,
    multiplier_ptr,
    out_ptr,
    n_elements,
    n_channels,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index for each element (assuming NCDHW format)
    channel_offsets = (offsets // (n_elements // n_channels)) % n_channels
    multiplier = tl.load(multiplier_ptr + channel_offsets, mask=mask, other=1.0)
    
    # First LeakyReLU
    x = tl.where(x > 0, x, x * negative_slope)
    
    # Multiply
    x = x * multiplier
    
    # Second LeakyReLU
    x = tl.where(x > 0, x, x * negative_slope)
    
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_leaky_mul_leaky(x: torch.Tensor, multiplier: torch.Tensor, negative_slope: float = 0.2):
    assert x.is_cuda and multiplier.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    multiplier = multiplier.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    n_channels = x.shape[1]
    BLOCK_SIZE = 1024
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_leaky_mul_leaky_kernel[grid](
        x, multiplier, out, n_elements, n_channels,
        negative_slope=negative_slope,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def max_pool3d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_depth,
    in_height,
    in_width,
    out_depth,
    out_height,
    out_width,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)
    
    out_d = pid_d * BLOCK_D
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    
    if out_d >= out_depth or out_h >= out_height or out_w >= out_width:
        return
    
    max_val = -1e10
    
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                in_d = out_d * stride_d + kd
                in_h = out_h * stride_h + kh
                in_w = out_w * stride_w + kw
                
                if in_d < in_depth and in_h < in_height and in_w < in_width:
                    in_offset = (
                        pid_b * channels * in_depth * in_height * in_width +
                        pid_c * in_depth * in_height * in_width +
                        in_d * in_height * in_width +
                        in_h * in_width +
          
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 74
```
