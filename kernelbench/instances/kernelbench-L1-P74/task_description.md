# KernelBench Level 1 Problem 74: 74_conv_transposed_1D_dilated.py

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
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

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
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose(x)

# Test code
batch_size = 32
in_channels = 32
out_channels = 64
kernel_size = 5
length = 131072
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
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

A previous single-shot attempt failed. Error: `PassManager::run failed`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_fused_kernel(
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
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation,
    # Strides for input (N, C, L)
    input_batch_stride,
    input_channel_stride,
    input_length_stride,
    # Strides for weight (in_c, out_c, k)
    weight_in_channel_stride,
    weight_out_channel_stride,
    weight_kernel_stride,
    # Strides for output (N, C, L)
    output_batch_stride,
    output_channel_stride,
    output_length_stride,
    # Config
    HAS_BIAS: tl.constexpr,
    BLOCK_LENGTH: tl.constexpr,
    BLOCK_OUT_CHANNELS: tl.constexpr,
    BLOCK_IN_CHANNELS: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_length = tl.program_id(1)
    pid_out_channel_block = tl.program_id(2)
    
    # Output positions this block handles
    out_pos = pid_length * BLOCK_LENGTH + tl.arange(0, BLOCK_LENGTH)
    out_channel_start = pid_out_channel_block * BLOCK_OUT_CHANNELS
    out_channels_range = out_channel_start + tl.arange(0, BLOCK_OUT_CHANNELS)
    
    # Masks
    length_mask = out_pos < output_length
    channel_mask = out_channels_range < out_channels
    mask = length_mask[:, None] & channel_mask[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_LENGTH, BLOCK_OUT_CHANNELS), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for ic_block_start in range(0, in_channels, BLOCK_IN_CHANNELS):
        ic_range = ic_block_start + tl.arange(0, BLOCK_IN_CHANNELS)
        ic_mask = ic_range < in_channels
        
        # Loop over kernel positions
        for k in range(kernel_size):
            # For transposed conv: output_pos = input_pos * stride - padding + k * dilation
            # Rearranged: input_pos = (output_pos + padding - k * dilation) / stride
            # Input position must be integer and valid
            
            # Calculate input positions for all output positions
            input_pos_numer = out_pos + padding - k * dilation
            input_pos = input_pos_numer // stride
            
            # Check if division is exact (no remainder)
            valid_division = (input_pos_numer % stride) == 0
            valid_input = valid_division[:, None] & (input_pos[:, None] >= 0) & (input_pos[:, None] < input_length)
            
            # Load weights for this kernel position: shape (BLOCK_IN_CHANNELS, BLOCK_OUT_CHANNELS)
            weight_idx = (ic_range[:, None] * weight_in_channel_stride + 
                         out_channels_range[None, :] * weight_out_channel_stride + 
                         k * weight_kernel_stride)
            w_mask = ic_mask[:, None] & channel_mask[None, :]
            w = tl.load(weight_ptr + weight_idx, ma
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 74
```
