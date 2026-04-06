# KernelBench Level 1 Problem 73: 73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped.py

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
    Performs a 3D transposed convolution operation with asymmetric input and square kernel.
    The input is padded before the convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 4
in_channels = 32
out_channels = 32
kernel_size = 3
depth = 32
height = 64
width = 128
stride = 2
padding = 1
groups = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups]
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

A previous single-shot attempt failed. Error: `at 40:16:
    output_batch_stride,
    output_channel_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    has_bias: tl.constexpr,
    BLOCK_OUT_CHANNEL: tl.const...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    in_depth,
    in_height,
    in_width,
    out_depth,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    groups,
    input_batch_stride,
    input_channel_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    has_bias: tl.constexpr,
    BLOCK_OUT_CHANNEL: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_out_d = tl.program_id(1)
    pid_out_h = tl.program_id(2)
    pid_out_w = tl.program_id(3)
    pid_oc_block = tl.program_id(4)

    oc_start = pid_oc_block * BLOCK_OUT_CHANNEL
    oc_end = tl.minimum(oc_start + BLOCK_OUT_CHANNEL, out_channels)

    channels_per_group = out_channels // groups
    in_channels_per_group = in_channels // groups

    for oc_idx in range(oc_end - oc_start):
        oc = oc_start + oc_idx
        group_idx = oc // channels_per_group
        ic_start = group_idx * in_channels_per_group
        ic_end = ic_start + in_channels_per_group

        acc = 0.0

        for ic in range(ic_start, ic_end):
            for kd in range(kernel_size):
                for kh in range(kernel_size):
                    for kw in range(kernel_size):
                        in_d = pid_out_d * stride - padding + kd
                        in_h = pid_out_h * stride - padding + kh
                        in_w = pid_out_w * stride - padding + kw

                        if (in_d >= 0) & (in_d < in_depth) & \
                           (in_h >= 0) & (in_h < in_height) & \
                           (in_w >= 0) & (in_w < in_width):
                            input_offset = (
                                pid_batch * input_batch_stride +
                                ic * input_channel_stride +
                                in_d * input_depth_stride +
                                in_h * input_height_stride +
                                in_w * input_width_stride
                            )
                            input_val = tl.load(input_ptr + input_offset)

                            weight_offset = (
                                oc * weight_out_channel_stride +
                                ic * weight_in_channel_stride +
                                kd * weight_depth_stride +
                                kh * weight_height_stride +
                                kw * weight_width_stride
                            )
                            weight_val = tl.load(weight_ptr + weight_offset)

                            acc
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 73
```
