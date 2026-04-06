# KernelBench Level 2 Problem 1: 1_Conv2D_ReLU_BiasAdd.py

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
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = x + self.bias
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
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

A previous single-shot attempt failed. Error: `at 47:22:
    BLOCK_SIZE_HEIGHT: tl.constexpr,
    BLOCK_SIZE_WIDTH: tl.constexpr,
    BLOCK_SIZE_IN_CHANNEL: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    PADDING: tl.constexpr,
    STRIDE: tl....`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_relu_bias_kernel(
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
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_size,
    # Strides for input tensor (NCHW)
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    # Strides for weight tensor (OCHW)
    weight_out_channel_stride,
    weight_in_channel_stride,
    weight_height_stride,
    weight_width_stride,
    # Strides for output tensor (NCHW)
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    # Bias stride
    bias_stride,
    # Block sizes
    BLOCK_SIZE_OUT_CHANNEL: tl.constexpr,
    BLOCK_SIZE_HEIGHT: tl.constexpr,
    BLOCK_SIZE_WIDTH: tl.constexpr,
    BLOCK_SIZE_IN_CHANNEL: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    PADDING: tl.constexpr,
    STRIDE: tl.constexpr,
):
    # Program IDs for output dimensions
    pid_batch = tl.program_id(0)
    pid_out_channel_block = tl.program_id(1)
    pid_height_block = tl.program_id(2)
    pid_width_block = tl.program_id(3)
    
    # Calculate output positions
    out_channel_start = pid_out_channel_block * BLOCK_SIZE_OUT_CHANNEL
    height_start = pid_height_block * BLOCK_SIZE_HEIGHT
    width_start = pid_width_block * BLOCK_SIZE_WIDTH
    
    # Create ranges for output channels, height, width
    out_channel_offsets = out_channel_start + tl.arange(0, BLOCK_SIZE_OUT_CHANNEL)
    height_offsets = height_start + tl.arange(0, BLOCK_SIZE_HEIGHT)
    width_offsets = width_start + tl.arange(0, BLOCK_SIZE_WIDTH)
    
    # Masks for valid output positions
    out_channel_mask = out_channel_offsets < out_channels
    height_mask = height_offsets < out_height
    width_mask = width_offsets < out_width
    
    # 3D mask for output
    output_mask = out_channel_mask[:, None, None] & height_mask[None, :, None] & width_mask[None, None, :]
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_OUT_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH), dtype=tl.float32)
    
    # Loop over input channels
    for ic in range(0, in_channels, BLOCK_SIZE_IN_CHANNEL):
        in_channel_offsets = ic + tl.arange(0, BLOCK_SIZE_IN_CHANNEL)
        in_channel_mask = in_channel_offsets < in_channels
        
        # Loop over kernel height and width
        for kh in range(KERNEL_SIZE):
            for kw in range(KERNEL_SIZE):
                # Calculate input positions (with padding and stride)
                in_height_start = height_start * STRIDE - PADDING + kh
                in_width_start = width_start * STRIDE - PADDING + kw
                
                # Create input position offsets
                in_height_offsets = in_height_start + tl.arange(0, BLOCK_SIZ
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 1
```
