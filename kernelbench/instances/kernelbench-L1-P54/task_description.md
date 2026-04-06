# KernelBench Level 1 Problem 54: 54_conv_standard_3D__square_input__square_kernel.py

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
    Performs a standard 3D convolution operation with square input and square kernel.

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
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv3d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
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

A previous single-shot attempt failed. The output was incorrect.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Input pointers
    input_ptr,
    weight_ptr,
    bias_ptr,
    # Output pointer
    output_ptr,
    # Dimensions
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
    dilation,
    groups,
    has_bias,
    # Strides for input tensor (N, C, D, H, W)
    input_batch_stride,
    input_channel_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    # Strides for weight tensor (C_out, C_in/groups, kD, kH, kW)
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    # Strides for output tensor (N, C, D, H, W)
    output_batch_stride,
    output_channel_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    # BLOCK_SIZE for loop unrolling
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one output element (n, c_out, d_out, h_out, w_out)
    pid = tl.program_id(0)
    
    # Decode the output position from linear index
    tmp = pid
    w_out = tmp % out_width
    tmp = tmp // out_width
    h_out = tmp % out_height
    tmp = tmp // out_height
    d_out = tmp % out_depth
    tmp = tmp // out_depth
    c_out = tmp % out_channels
    n = tmp // out_channels
    
    # Compute input starting position (top-left corner of receptive field)
    d_in_start = d_out * stride - padding
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding
    
    # Determine which group this output channel belongs to
    channels_per_group = out_channels // groups
    in_channels_per_group = in_channels // groups
    group_idx = c_out // channels_per_group
    c_in_start = group_idx * in_channels_per_group
    
    # Accumulate convolution result
    acc = 0.0
    
    # Iterate through input channels in this group
    for c_in_range in range(in_channels_per_group):
        c_in = c_in_start + c_in_range
        
        # Iterate through kernel dimensions
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Calculate input coordinates with dilation
                    d_in = d_in_start + kd * dilation
                    h_in = h_in_start + kh * dilation
                    w_in = w_in_start + kw * dilation
                    
                    # Check bounds - only compute if within input bounds
                    if (d_in >= 0) and (d_in < in_depth) and \
                       (h_in >= 0) and (h_in < in_height) and \
                       (w_in >= 0) and (w_in < in_width):
                        # Load input value
                        input_offset = (n * input_batch_stride + 
                                      
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 54
```
