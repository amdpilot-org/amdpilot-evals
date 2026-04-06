# KernelBench Level 1 Problem 62: 62_conv_standard_2D__square_input__asymmetric_kernel.py

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
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 8
in_channels = 32
out_channels = 64
kernel_size = (5, 9)
width = 512
height = 512

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
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

A previous single-shot attempt failed. Error: `at 60:12:
    group_id = out_c // out_channels_per_group

    acc = tl.zeros([BLOCK_SIZE_OC], dtype=tl.float32)

    for ic_block in range(0, channels_per_group, BLOCK_SIZE_IC):
        ic_offsets ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    kernel_height,
    kernel_width,
    out_height,
    out_width,
    stride,
    padding,
    dilation,
    groups,
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    w_out_channel_stride,
    w_in_channel_stride,
    w_height_stride,
    w_width_stride,
    out_batch_stride,
    out_channel_stride,
    out_height_stride,
    out_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    pid = tl.program_id(0)
    
    out_n = pid // (out_channels * out_height * out_width)
    remainder = pid % (out_channels * out_height * out_width)
    out_c = remainder // (out_height * out_width)
    remainder = remainder % (out_height * out_width)
    out_h = remainder // out_width
    out_w = remainder % out_width
    
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    group_id = out_c // out_channels_per_group
    
    acc = tl.zeros([BLOCK_SIZE_OC], dtype=tl.float32)
    
    for ic_block in range(0, channels_per_group, BLOCK_SIZE_IC):
        ic_offsets = ic_block + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < channels_per_group
        
        in_c_base = group_id * channels_per_group + ic_offsets
        
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                in_h = in_h_start + kh * dilation
                in_w = in_w_start + kw * dilation
                
                valid_h = (in_h >= 0) & (in_h < in_height)
                valid_w = (in_w >= 0) & (in_w < in_width)
                valid_spatial = valid_h & valid_w
                
                x_offsets = (
                    out_n * x_batch_stride +
                    in_c_base * x_channel_stride +
                    in_h * x_height_stride +
                    in_w * x_width_stride
                )
                
                x_vals = tl.load(
                    x_ptr + x_offsets,
                    mask=ic_mask & valid_spatial,
                    other=0.0
                )
                
                w_offsets = (
                    out_c * w_out_channel_stride +
                    in_c_base * w_in_channel_stride +
                    kh * w_height_stride +
                    kw * w_width_stride
                )
                
                w_vals = tl.load(
                    weight_ptr + w_offsets,
                    mask=ic_mask,
                    other=0.0
                )
                
                acc += x_vals * w_vals
    
    acc_sum = tl.sum(acc, axis=0)
    
    if HAS_BIAS:
      
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 62
```
