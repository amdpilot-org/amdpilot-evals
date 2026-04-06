# KernelBench Level 1 Problem 85: 85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel.py

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
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 32
in_channels = 128
out_channels = 128
kernel_size_h = 3
kernel_size_w = 7
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]
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

A previous single-shot attempt failed. Error: `at 42:12:
    output_height_stride,
    output_width_stride,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    NUM_KERNEL_ELEMENTS: tl.constexpr,
):
    # Progra...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    output_ptr,
    # Dimensions
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    # Strides
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_channel_stride,
    weight_kernel_h_stride,
    weight_kernel_w_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    # Block sizes
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    NUM_KERNEL_ELEMENTS: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel
    pid_h = tl.program_id(2)  # output height block
    pid_w = tl.program_id(3)  # output width block
    
    # Calculate output starting positions for this block
    out_h_start = pid_h * BLOCK_SIZE_H
    out_w_start = pid_w * BLOCK_SIZE_W
    
    # Create output position offsets
    out_h_offsets = out_h_start + tl.arange(0, BLOCK_SIZE_H)
    out_w_offsets = out_w_start + tl.arange(0, BLOCK_SIZE_W)
    
    # Create 2D meshgrid for output positions
    out_h_grid, out_w_grid = tl.meshgrid(out_h_offsets, out_w_offsets, indexing='ij')
    
    # Mask for valid output positions
    out_h_mask = out_h_grid < out_height
    out_w_mask = out_w_grid < out_width
    out_mask = out_h_mask & out_w_mask
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_H, BLOCK_SIZE_W], dtype=tl.float32)
    
    # Loop over kernel elements
    for k_idx in range(NUM_KERNEL_ELEMENTS):
        # Calculate kernel position from linear index
        kh = k_idx // kernel_w
        kw = k_idx % kernel_w
        
        # Calculate corresponding input positions
        in_h = out_h_grid * stride_h - padding_h + kh * dilation_h
        in_w = out_w_grid * stride_w - padding_w + kw * dilation_w
        
        # Check input bounds
        in_h_mask = (in_h >= 0) & (in_h < in_height)
        in_w_mask = (in_w >= 0) & (in_w < in_width)
        load_mask = out_mask & in_h_mask & in_w_mask
        
        # Calculate input memory offsets
        input_offset = (pid_b * input_batch_stride + 
                       pid_c * input_channel_stride +
                       in_h * input_height_stride +
                       in_w * input_width_stride)
        
        # Load input values
        input_vals = tl.load(input_ptr + input_offset, mask=load_mask, other=0.0)
        
        # Load weight value for this kernel position and channel
        weight_offset = (pid_c * weight_channel_stride + 
                        kh * weight_kernel_h_stride + 
                        kw * weight_kernel_w_stride)
        weight_val = tl.l
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 85
```
