# KernelBench Level 2 Problem 67: 67_Conv2d_GELU_GlobalAvgPool.py

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
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        x = self.conv(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
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

A previous single-shot attempt failed. Error: `at 41:19:

        x_offsets = (batch_idx * stride_b + 
                    channel_idx * stride_c + 
                    h_idx * stride_h + 
                    w_idx * stride_w)

        x = tl.l...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def gelu_global_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    n_spatial = height * width
    acc = 0.0
    
    for start in range(0, n_spatial, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        
        h_idx = offsets // width
        w_idx = offsets % width
        
        x_offsets = (batch_idx * stride_b + 
                    channel_idx * stride_c + 
                    h_idx * stride_h + 
                    w_idx * stride_w)
        
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x3 = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x3)
        tanh_val = tl.libdevice.tanh(inner)
        gelu_x = 0.5 * x * (1.0 + tanh_val)
        
        acc += tl.sum(gelu_x * mask.to(tl.float32))
    
    out_val = acc / n_spatial
    tl.store(out_ptr + pid, out_val)


def gelu_global_pool(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    out = torch.empty(batch_size, channels, dtype=x.dtype, device=x.device)
    
    stride_b, stride_c, stride_h, stride_w = x.stride()
    
    n_outputs = batch_size * channels
    BLOCK_SIZE = 256
    
    grid = (n_outputs,)
    
    gelu_global_pool_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        stride_b, stride_c, stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = gelu_global_pool(x)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 67
```
