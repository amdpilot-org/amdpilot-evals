# KernelBench Level 2 Problem 89: 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max.py

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
    A model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels)) # Assuming subtraction is element-wise across channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = torch.softmax(x, dim=1) # Apply softmax across channels (dim=1)
        x = x - self.subtract.view(1, -1, 1, 1, 1) # Subtract across channels
        x = torch.sigmoid(x) * x # Swish activation
        x = torch.max(x, dim=1)[0] # Max pooling across channels
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]
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

A previous single-shot attempt failed. Error: `at 86:4:

        # Swish: sigmoid(x) * x
        sigmoid_val = 1.0 / (1.0 + tl.exp(-val))
        swish_val = sigmoid_val * val

        # Track max across channels
        block_max = tl.max(swis...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_subtract_swish_max_kernel(
    x_ptr,           # Input tensor pointer [B, C, D, H, W]
    subtract_ptr,    # Subtract parameter pointer [C]
    out_ptr,         # Output tensor pointer [B, D, H, W]
    batch_size,
    channels,
    depth,
    height,
    width,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial position (b, d, h, w)
    # We process all channels for that position and output the max across channels
    
    pid = tl.program_id(0)
    
    # Decode the spatial position from pid
    # Total spatial positions = batch_size * depth * height * width
    spatial_id = pid
    
    w_idx = spatial_id % width
    spatial_id = spatial_id // width
    h_idx = spatial_id % height
    spatial_id = spatial_id // height
    d_idx = spatial_id % depth
    b_idx = spatial_id // depth
    
    # Compute base offset for this spatial position
    base_offset = b_idx * stride_b + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
    
    # First pass: compute max for numerical stability in softmax
    max_val = tl.zeros([1], dtype=tl.float32) - 1e10
    for channel_start in range(0, channels, BLOCK_SIZE):
        channel_offsets = channel_start + tl.arange(0, BLOCK_SIZE)
        mask = channel_offsets < channels
        offsets = base_offset + channel_offsets * stride_c
        x = tl.load(x_ptr + offsets, mask=mask, other=-1e10)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute softmax, subtract, swish, and track max
    sum_exp = tl.zeros([1], dtype=tl.float32)
    
    for channel_start in range(0, channels, BLOCK_SIZE):
        channel_offsets = channel_start + tl.arange(0, BLOCK_SIZE)
        mask = channel_offsets < channels
        offsets = base_offset + channel_offsets * stride_c
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Softmax with numerical stability
        x_exp = tl.exp(x - max_val)
        sum_exp += tl.sum(x_exp * mask.to(tl.float32), axis=0)
    
    # Third pass: compute final values and max
    max_out = tl.zeros([1], dtype=tl.float32) - 1e10
    
    for channel_start in range(0, channels, BLOCK_SIZE):
        channel_offsets = channel_start + tl.arange(0, BLOCK_SIZE)
        mask = channel_offsets < channels
        offsets = base_offset + channel_offsets * stride_c
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        subtract_val = tl.load(subtract_ptr + channel_offsets, mask=mask, other=0.0)
        
        # Softmax
        x_exp = tl.exp(x - max_val)
        softmax_val = x_exp / sum_exp
        
        # Subtract
        val = softmax_val - subtract_val
        
        # Swish: sigmoid(x) * x
        sigmoid_val = 1.0 / (1.0 + tl.exp(-val))
        swish_val = sigmoid_val
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 89
```
