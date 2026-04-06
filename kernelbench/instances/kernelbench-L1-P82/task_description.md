# KernelBench Level 1 Problem 82: 82_conv_depthwise_2D_square_input_square_kernel.py

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
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 16
in_channels = 64
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
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
    kernel_size,
    stride,
    padding,
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    w_channel_stride,
    w_kh_stride,
    w_kw_stride,
    out_batch_stride,
    out_channel_stride,
    out_height_stride,
    out_width_stride,
    HAS_BIAS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    out_width_idx = pid % out_width
    pid = pid // out_width
    out_height_idx = pid % out_height
    pid = pid // out_height
    channel_idx = pid % in_channels
    batch_idx = pid // in_channels
    
    in_height_start = out_height_idx * stride - padding
    in_width_start = out_width_idx * stride - padding
    
    acc = 0.0
    
    for kh in range(KERNEL_SIZE):
        for kw in range(KERNEL_SIZE):
            in_h = in_height_start + kh
            in_w = in_width_start + kw
            
            if in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width:
                x_offset = (
                    batch_idx * x_batch_stride +
                    channel_idx * x_channel_stride +
                    in_h * x_height_stride +
                    in_w * x_width_stride
                )
                x_val = tl.load(x_ptr + x_offset)
                
                w_offset = (
                    channel_idx * w_channel_stride +
                    kh * w_kh_stride +
                    kw * w_kw_stride
                )
                w_val = tl.load(weight_ptr + w_offset)
                
                acc += x_val * w_val
    
    if HAS_BIAS:
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val
    
    out_offset = (
        batch_idx * out_batch_stride +
        channel_idx * out_channel_stride +
        out_height_idx * out_height_stride +
        out_width_idx * out_width_stride
    )
    tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    batch_size, in_channels, in_height, in_width = x.shape
    kernel_size = weight.shape[2]
    
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty(
        batch_size, in_channels, out_height, out_width,
        dtype=x.dtype, device=x.device
    )
    
    x_batch_stride = x.stride(0)
    x_channel_stride = x.stride(1)
    x_height_stride = x.strid
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 82
```
