# KernelBench Level 1 Problem 86: 86_conv_depthwise_separable_2D.py

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
    Performs a depthwise-separable 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Test code
batch_size = 16
in_channels = 64
out_channels = 128
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
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
def depthwise_conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    stride,
    padding,
    dilation,
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    w_channel_stride,
    w_kernel_h_stride,
    w_kernel_w_stride,
    out_batch_stride,
    out_channel_stride,
    out_height_stride,
    out_width_stride,
    HAS_BIAS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    batch_idx = pid_bc // in_channels
    channel_idx = pid_bc % in_channels

    out_h_start = pid_h * BLOCK_H
    out_w_start = pid_w * BLOCK_W

    out_h_offs = out_h_start + tl.arange(0, BLOCK_H)
    out_w_offs = out_w_start + tl.arange(0, BLOCK_W)

    out_h_grid = out_h_offs[:, None]
    out_w_grid = out_w_offs[None, :]

    out_h_mask = out_h_grid < out_height
    out_w_mask = out_w_grid < out_width
    out_mask = out_h_mask & out_w_mask

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for kh in tl.static_range(KERNEL_SIZE):
        for kw in tl.static_range(KERNEL_SIZE):
            in_h = out_h_grid * stride - padding + kh * dilation
            in_w = out_w_grid * stride - padding + kw * dilation

            in_h_mask = (in_h >= 0) & (in_h < in_height)
            in_w_mask = (in_w >= 0) & (in_w < in_width)
            mask = in_h_mask & in_w_mask & out_mask

            in_h_clamped = tl.maximum(tl.minimum(in_h, in_height - 1), 0)
            in_w_clamped = tl.maximum(tl.minimum(in_w, in_width - 1), 0)

            x_offs = (batch_idx * x_batch_stride +
                      channel_idx * x_channel_stride +
                      in_h_clamped * x_height_stride +
                      in_w_clamped * x_width_stride)
            x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

            w_offs = (channel_idx * w_channel_stride +
                      kh * w_kernel_h_stride +
                      kw * w_kernel_w_stride)
            w = tl.load(weight_ptr + w_offs)

            acc += x * w

    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel_idx)
        acc += bias

    out_offs = (batch_idx * out_batch_stride +
                channel_idx * out_channel_stride +
                out_h_grid * out_height_stride +
                out_w_grid * out_width_stride)
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    w_out_channel_stride,
    w_in_channel_stride,
    out_batch_stride,
    out_chann
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 86
```
