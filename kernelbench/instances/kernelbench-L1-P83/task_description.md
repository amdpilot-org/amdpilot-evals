# KernelBench Level 1 Problem 83: 83_conv_depthwise_2D_square_input_asymmetric_kernel.py

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
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        
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
batch_size = 64
in_channels = 8
kernel_size = 3
width = 512
height = 512
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]
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

A previous single-shot attempt failed. Error: `'function' object has no attribute 'constexpr'`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    # Pointers to inputs
    x_ptr,
    weight_ptr,
    bias_ptr,
    # Pointer to output
    out_ptr,
    # Dimensions
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    height_in: tl.constexpr,
    width_in: tl.constexpr,
    height_out: tl.constexpr,
    width_out: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    has_bias: tl.constexpr,
    # Strides for memory access
    stride_x_batch: tl.constexpr,
    stride_x_channel: tl.constexpr,
    stride_x_height: tl.constexpr,
    stride_x_width: tl.constexpr,
    stride_out_batch: tl.constexpr,
    stride_out_channel: tl.constexpr,
    stride_out_height: tl.constexpr,
    stride_out_width: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate output position from pid
    # pid maps to (batch, channel, height_out, width_out)
    width_out_idx = pid % width_out
    pid_temp = pid // width_out
    height_out_idx = pid_temp % height_out
    pid_temp = pid_temp // height_out
    channel_idx = pid_temp % in_channels
    batch_idx = pid_temp // in_channels
    
    # Calculate corresponding input height start position
    height_in_start = height_out_idx * stride - padding
    
    # Accumulator for the convolution result
    acc = 0.0
    
    # Loop over kernel height dimension
    for k in range(kernel_size):
        # Calculate input height position with dilation
        h_in = height_in_start + k * dilation
        
        # Check bounds
        if h_in >= 0 and h_in < height_in:
            # Calculate input offset
            x_offset = (
                batch_idx * stride_x_batch +
                channel_idx * stride_x_channel +
                h_in * stride_x_height +
                width_out_idx * stride_x_width
            )
            
            # Load input value
            x_val = tl.load(x_ptr + x_offset)
            
            # Load weight value (depthwise: one weight per channel per kernel position)
            w_offset = channel_idx * kernel_size + k
            w_val = tl.load(weight_ptr + w_offset)
            
            # Accumulate
            acc += x_val * w_val
    
    # Add bias if present
    if has_bias:
        bias_val = tl.load(bias_ptr + channel_idx)
        acc += bias_val
    
    # Calculate output offset
    out_offset = (
        batch_idx * stride_out_batch +
        channel_idx * stride_out_channel +
        height_out_idx * stride_out_height +
        width_out_idx * stride_out_width
    )
    
    # Store output
    tl.store(out_ptr + out_offset, acc)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.T
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 83
```
