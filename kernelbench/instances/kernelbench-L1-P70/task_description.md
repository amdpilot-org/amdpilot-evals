# KernelBench Level 1 Problem 70: 70_conv_transposed_3D__asymmetric_input__square_kernel.py

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
    Performs a transposed 3D convolution operation with asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of each dimension in the output shape. 
                                                  Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), 
                                                stride=stride, padding=padding, output_padding=output_padding, 
                                                dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 8
in_channels = 48
out_channels = 24
kernel_size = 3
depth = 96
height = 96
width = 96

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, height, width)
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

A previous single-shot attempt failed. Error: `at 51:20:
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    # Block sizes
    BLOCK_OUT_CHANNEL: tl.constexpr,
    BLOCK_KERNEL: tl.constexpr,
):
    # Program IDs for...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    bias_ptr,
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
    # Conv parameters
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    has_bias,
    # Strides - input (NCDHW)
    input_batch_stride,
    input_channel_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    # Strides - output (NCDHW)
    output_batch_stride,
    output_channel_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    # Strides - weight (in_c, out_c/g, kD, kH, kW)
    weight_in_channel_stride,
    weight_out_channel_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    # Block sizes
    BLOCK_OUT_CHANNEL: tl.constexpr,
    BLOCK_KERNEL: tl.constexpr,
):
    # Program IDs for output position
    pid_batch = tl.program_id(0)
    pid_out_depth = tl.program_id(1)
    pid_out_height = tl.program_id(2)
    pid_out_width = tl.program_id(3)
    pid_out_channel_block = tl.program_id(4)
    
    # Output channel indices for this block
    out_c_base = pid_out_channel_block * BLOCK_OUT_CHANNEL
    out_c_offsets = out_c_base + tl.arange(0, BLOCK_OUT_CHANNEL)
    out_c_mask = out_c_offsets < out_channels
    
    # Output spatial indices
    out_d = pid_out_depth
    out_h = pid_out_height
    out_w = pid_out_width
    
    # Bounds check for spatial dimensions
    spatial_mask = (out_d < out_depth) & (out_h < out_height) & (out_w < out_width)
    
    # Initialize accumulator for output channels
    acc = tl.zeros((BLOCK_OUT_CHANNEL,), dtype=tl.float32)
    
    # For transposed convolution, iterate over input channels and kernel positions
    # Each output position receives contributions from multiple input positions
    
    # Calculate which input channels this output channel group depends on
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    for g in range(groups):
        # Input channel range for this group
        in_c_start = g * channels_per_group
        in_c_end = in_c_start + channels_per_group
        
        # Output channel range for this group
        out_c_group_start = g * out_channels_per_group
        out_c_group_end = out_c_group_start + out_channels_per_group
        
        # Check if any output channels in this block belong to this group
        group_mask = (out_c_offsets >= out_c_group_start) & (out_c_offsets < out_c_group_end)
        
        if tl.sum(group_mask) == 0:
            continue
        
        # Iterate over input channels in this group
        for in_c_idx in range(channels_per_group):
            in_c = in_c_start + in_c_
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 70
```
