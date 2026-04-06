# KernelBench Level 1 Problem 75: 75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__.py

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
    Performs a 2D transposed convolution operation with asymmetric input, asymmetric kernel, 
    grouped, padded, and dilated.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        dilation (tuple, optional): Spacing between kernel elements (height, width). Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)
height = 128
width = 256
stride = (2, 3)
padding = (1, 2)
dilation = (2, 1)
groups = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, groups]
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

A previous single-shot attempt failed. Error: `at 41:16:
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_OUT_H: tl.constexpr,
    BLOCK_OUT_W: tl.const...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    padding_height,
    padding_width,
    dilation_height,
    dilation_width,
    groups,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_kernel_height_stride,
    weight_kernel_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_OUT_H: tl.constexpr,
    BLOCK_OUT_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_out_h = tl.program_id(1)
    pid_out_w = tl.program_id(2)
    pid_out_c = tl.program_id(3)

    out_h_start = pid_out_h * BLOCK_OUT_H
    out_w_start = pid_out_w * BLOCK_OUT_W

    out_h_off = out_h_start + tl.arange(0, BLOCK_OUT_H)
    out_w_off = out_w_start + tl.arange(0, BLOCK_OUT_W)

    mask_h = out_h_off < out_height
    mask_w = out_w_off < out_width
    mask_hw = mask_h[:, None] & mask_w[None, :]

    group_size_out = out_channels // groups
    group_size_in = in_channels // groups
    group_idx = pid_out_c // group_size_out
    in_c_start = group_idx * group_size_in
    local_out_c = pid_out_c - group_idx * group_size_out

    acc = tl.zeros((BLOCK_OUT_H, BLOCK_OUT_W), dtype=tl.float32)

    for in_c_local in range(group_size_in):
        in_c = in_c_start + in_c_local

        for kh in range(kernel_height):
            for kw in range(kernel_width):
                in_h = out_h_off * stride_height - padding_height + kh * dilation_height
                in_w = out_w_off * stride_width - padding_width + kw * dilation_width

                mask_in_h = (in_h >= 0) & (in_h < in_height)
                mask_in_w = (in_w >= 0) & (in_w < in_width)
                mask_in = mask_in_h[:, None] & mask_in_w[None, :] & mask_hw

                in_h_clamped = tl.maximum(tl.minimum(in_h, in_height - 1), 0)
                in_w_clamped = tl.maximum(tl.minimum(in_w, in_width - 1), 0)

                input_offset = (
                    pid_batch * input_batch_stride +
                    in_c * input_channel_stride +
                    in_h_clamped * input_height_stride +
                    in_w_clamped * input_width_stride
                )
                input_val = tl.load(input_ptr + input_offset, mask=mask_in, other=0.0)

                weight_offset = (
                    local_out_c * weight_out_channel_stride +
                    in_c_local * weight_in_channel_stride +
                    kh * weight_kernel_height_stride +
                    kw * weight_kernel_width_stride
                )
         
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 75
```
