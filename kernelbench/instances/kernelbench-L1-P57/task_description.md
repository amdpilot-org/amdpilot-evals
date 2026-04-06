# KernelBench Level 1 Problem 57: 57_conv_transposed_2D__square_input__square_kernel.py

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
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv_transpose2d(x)

# Test code
batch_size = 8
in_channels = 64  # double channels for heavier compute
out_channels = 64
kernel_size = 3
# larger square input
height = 1024
width = 1024

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

A previous single-shot attempt failed. Error: `at 37:12:
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.con...`

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
    kernel_size,
    stride,
    padding,
    output_padding,
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
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_out_channel = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    out_h_start = pid_h * BLOCK_SIZE_H
    out_w_start = pid_w * BLOCK_SIZE_W

    out_h_offs = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_offs = out_w_start + tl.arange(0, BLOCK_SIZE_W)

    mask_h = out_h_offs < out_height
    mask_w = out_w_offs < out_width
    mask = mask_h[:, None] & mask_w[None, :]

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    in_channel_per_group = in_channels // groups
    out_channel_per_group = out_channels // groups
    group_id = pid_out_channel // out_channel_per_group
    in_channel_start = group_id * in_channel_per_group

    for kh in range(kernel_size):
        for kw in range(kernel_size):
            in_h_num = out_h_offs[:, None] - kh + padding
            in_w_num = out_w_offs[None, :] - kw + padding

            valid_h = (in_h_num >= 0) & (in_h_num < in_height * stride) & ((in_h_num % stride) == 0)
            valid_w = (in_w_num >= 0) & (in_w_num < in_width * stride) & ((in_w_num % stride) == 0)
            valid = valid_h & valid_w

            in_h = in_h_num // stride
            in_w = in_w_num // stride

            for ic_offset in range(in_channel_per_group):
                in_channel = in_channel_start + ic_offset

                weight_idx = (
                    pid_out_channel * weight_out_channel_stride +
                    in_channel * weight_in_channel_stride +
                    kh * weight_kernel_height_stride +
                    kw * weight_kernel_width_stride
                )
                w = tl.load(weight_ptr + weight_idx)

                input_idx = (
                    pid_batch * input_batch_stride +
                    in_channel * input_channel_stride +
                    in_h * input_height_stride +
                    in_w * input_width_stride
                )
                x = tl.load(input_ptr + input_idx, mask=valid, other=0.0)

                acc += x * w

    if HAS_BIAS:
        bias = tl.load(bias_ptr + pid_out_channel)
        acc += bias

    output_idx = (
        pid_batch * output_b
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 57
```
