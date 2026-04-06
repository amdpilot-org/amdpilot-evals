# KernelBench Level 1 Problem 63: 63_conv_standard_2D__square_input__square_kernel.py

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
    Performs a standard 2D convolution operation with a square input and square kernel.

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
batch_size = 16
in_channels = 16
out_channels = 128
kernel_size = 3
width = 1024
height = 1024

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

A previous single-shot attempt failed. Error: `at 26:19:
    padding,
    dilation,
    groups,
    has_bias: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.prog...`

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
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    has_bias: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)
    pid_oc_block = tl.program_id(3)

    oc_start = pid_oc_block * BLOCK_SIZE_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offsets < out_channels

    group_size_oc = out_channels // groups
    group_size_ic = in_channels // groups
    group_id = oc_start // group_size_oc
    ic_start = group_id * group_size_ic

    acc = tl.zeros((BLOCK_SIZE_OC,), dtype=tl.float32)

    ih_base = pid_oh * stride - padding
    iw_base = pid_ow * stride - padding

    for ic_block in range(0, group_size_ic, BLOCK_SIZE_IC):
        ic_offsets = ic_start + ic_block + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offsets < ic_start + group_size_ic

        for k_idx in range(kernel_size * kernel_size):
            kh = k_idx // kernel_size
            kw = k_idx % kernel_size
            ih = ih_base + kh * dilation
            iw = iw_base + kw * dilation

            spatial_valid = (ih >= 0) & (ih < in_height) & (iw >= 0) & (iw < in_width)

            x_idx = (
                pid_n * in_channels * in_height * in_width +
                ic_offsets * in_height * in_width +
                ih * in_width + iw
            )
            x = tl.load(x_ptr + x_idx, mask=ic_mask & spatial_valid, other=0.0)

            w_idx = (
                oc_offsets * group_size_ic * kernel_size * kernel_size +
                (ic_offsets - ic_start) * kernel_size * kernel_size +
                k_idx
            )
            w = tl.load(weight_ptr + w_idx, mask=oc_mask & ic_mask, other=0.0)

            acc += tl.sum(x * w, axis=0)

    if has_bias:
        bias = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += bias

    out_idx = (
        pid_n * out_channels * out_height * out_width +
        oc_offsets * out_height * out_width +
        pid_oh * out_width + pid_ow
    )
    tl.store(out_ptr + out_idx, acc, mask=oc_mask)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_height = (in_height + 2 * padding - dilation * (k
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 63
```
