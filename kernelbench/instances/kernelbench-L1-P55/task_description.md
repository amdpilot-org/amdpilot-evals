# KernelBench Level 1 Problem 55: 55_conv_standard_2D__asymmetric_input__square_kernel.py

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
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
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
# smaller spatial dims
height = 512
width = 1024
in_channels = 64  # increased channels
out_channels = 128
kernel_size = 3
# asymmetric input: make width considerably larger than height

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

A previous single-shot attempt failed. Error: `Evaluation timed out`

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
    height,
    width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    out_height,
    out_width,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_n = (batch_size * out_channels * out_height * out_width + BLOCK_N - 1) // BLOCK_N
    if pid >= num_blocks_n:
        return

    n_start = pid * BLOCK_N
    n_end = min(n_start + BLOCK_N, batch_size * out_channels * out_height * out_width)

    for idx in range(n_start, n_end):
        ow = idx % out_width
        tmp = idx // out_width
        oh = tmp % out_height
        tmp = tmp // out_height
        oc = tmp % out_channels
        n = tmp // out_channels

        ih_start = oh * stride - padding
        iw_start = ow * stride - padding

        in_channel_start = (oc // (out_channels // groups)) * (in_channels // groups)
        in_channel_end = in_channel_start + (in_channels // groups)

        acc = 0.0
        for ic in range(in_channel_start, in_channel_end):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    ih = ih_start + kh * dilation
                    iw = iw_start + kw * dilation
                    if ih >= 0 and ih < height and iw >= 0 and iw < width:
                        x_offset = (
                            n * x_batch_stride
                            + ic * x_channel_stride
                            + ih * x_height_stride
                            + iw * x_width_stride
                        )
                        x_val = tl.load(x_ptr + x_offset)
                        w_offset = (
                            oc * w_out_channel_stride
                            + ic * w_in_channel_stride
                            + kh * w_height_stride
                            + kw * w_width_stride
                        )
                        w_val = tl.load(weight_ptr + w_offset)
                        acc += x_val * w_val

        if HAS_BIAS:
            bias_val = tl.load(bias_ptr + oc)
            acc += bias_val

        out_offset = (
            n * out_batch_stride
            + oc * out_channel_stride
            + oh * out_height_stride
            + ow * out_width_stride
        )
        tl.store(out_ptr + out_offset, acc)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    gro
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 55
```
