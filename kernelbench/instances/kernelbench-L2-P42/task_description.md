# KernelBench Level 2 Problem 42: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply.py

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
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        x = x + self.bias
        x = torch.logsumexp(x, dim=1, keepdim=True)  # Log-sum-exp
        x = torch.sum(x, dim=(2, 3))  # Sum
        x = x * 10.0  # Multiplication
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height = width = 512
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

A previous single-shot attempt failed. The output was incorrect.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_mean_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    """
    Fused kernel: mean over spatial dims + bias.
    Each program handles one (batch, channel) pair.
    Input: (B, C, H, W)
    Output: (B, C)
    """
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    spatial_size = height * width
    
    # Compute mean over spatial dimensions using block processing
    sum_val = 0.0
    for block_start in range(0, spatial_size, BLOCK_SIZE_SPATIAL):
        offsets = block_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
        mask = offsets < spatial_size
        
        # Convert flat offset to 2D (h, w)
        h_indices = offsets // width
        w_indices = offsets % width
        
        # Compute memory offsets
        x_offsets = (
            batch_idx * stride_xb + 
            channel_idx * stride_xc + 
            h_indices * stride_xh + 
            w_indices * stride_xw
        )
        
        # Load values
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Sum
        sum_val += tl.sum(x_vals, axis=0)
    
    mean_val = sum_val / spatial_size
    
    # Add bias
    bias_val = tl.load(bias_ptr + channel_idx)
    result = mean_val + bias_val
    
    # Store output
    out_offset = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_offset, result)


@triton.jit
def logsumexp_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Logsumexp over channel dimension with numerical stability.
    Input: (B, C)
    Output: (B,)
    """
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    batch_idx = pid
    
    # First pass: find max across channels
    max_val = -1e10
    for c in range(0, channels, BLOCK_SIZE_C):
        offsets = c + tl.arange(0, BLOCK_SIZE_C)
        mask = offsets < channels
        
        x_offsets = batch_idx * channels + offsets
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=-1e10)
        
        block_max = tl.max(x_vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute sum of exp(x - max)
    sum_exp = 0.0
    for c in range(0, channels, BLOCK_SIZE_C):
        offsets = c + tl.arange(0, BLOCK_SIZE_C)
        mask = offsets < channels
        
        x_offsets = batch_idx * channels + offsets
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        exp_vals = tl.exp(x_vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)
    
    # Compute logsumexp
    result = max_val + tl.log(sum_exp)
  
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 42
```
