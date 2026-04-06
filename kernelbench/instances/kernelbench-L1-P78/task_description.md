# KernelBench Level 1 Problem 78: 78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__.py

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
    Performs a 2D transposed convolution operation with asymmetric input and kernel, with optional padding.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (tuple, optional): Stride of the convolution (height, width). Defaults to (1, 1).
        padding (tuple, optional): Padding applied to the input (height, width). Defaults to (0, 0).
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        
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
batch_size = 8
in_channels = 32
out_channels = 32
kernel_size = (3, 7)
height = 512
width = 1024
stride = (1, 1)
padding = (1, 3)

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

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

A previous single-shot attempt failed. Error: `at 39:16:
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_OUT_CHAN...`

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
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_OUT_CHANNEL: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_out_h = tl.program_id(1)
    pid_out_w = tl.program_id(2)
    pid_out_c = tl.program_id(3)

    out_h = pid_out_h * BLOCK_HEIGHT + tl.arange(0, BLOCK_HEIGHT)
    out_w = pid_out_w * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    out_c = pid_out_c * BLOCK_OUT_CHANNEL + tl.arange(0, BLOCK_OUT_CHANNEL)

    out_h_mask = out_h < out_height
    out_w_mask = out_w < out_width
    out_c_mask = out_c < out_channels

    acc = tl.zeros((BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_OUT_CHANNEL), dtype=tl.float32)

    for ic in range(in_channels):
        for kh in range(kernel_height):
            for kw in range(kernel_width):
                in_h_num = out_h + padding_height - kh
                in_w_num = out_w + padding_width - kw

                in_h_valid = (in_h_num % stride_height) == 0
                in_w_valid = (in_w_num % stride_width) == 0

                in_h = in_h_num // stride_height
                in_w = in_w_num // stride_width

                in_h_in_bounds = (in_h >= 0) & (in_h < in_height)
                in_w_in_bounds = (in_w >= 0) & (in_w < in_width)

                valid = in_h_valid & in_w_valid & in_h_in_bounds & in_w_in_bounds

                input_offset = (
                    pid_batch * input_batch_stride +
                    ic * input_channel_stride +
                    in_h * input_height_stride +
                    in_w * input_width_stride
                )
                input_val = tl.load(
                    input_ptr + input_offset,
                    mask=valid[:, None, None] & out_c_mask[None, None, :],
                    other=0.0
                )

                weight_offset = (
                    out_c * weight_out_channel_stride +
                    ic * weight_in_channel_stride +
                    kh * weight_height_stride +
                    kw * weight_width_stride
                )
                weight_val = tl.load(
                    weight_ptr + weight_offset,
                    mask=out_c_mask[None, None, :],
                    other=0.0
                )

                acc += in
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 78
```
