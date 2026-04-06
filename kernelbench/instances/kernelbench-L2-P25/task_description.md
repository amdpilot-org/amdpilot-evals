# KernelBench Level 2 Problem 25: 25_Conv2d_Min_Tanh_Tanh.py

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
    Model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.min(x, dim=1, keepdim=True)[0] # Apply minimum operation along the channel dimension
        x = torch.tanh(x)
        x = torch.tanh(x)
        return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
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

A previous single-shot attempt failed. Error: `at 40:12:

    # Calculate the pointer for each channel
    ptr = x_ptr + b * x_stride_0 + channel_offsets * x_stride_1 + h * x_stride_2 + w * x_stride_3

    # Load values
    vals = tl.load(ptr, ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def min_tanh_fused_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    x_stride_3,
    out_stride_0,
    out_stride_2,
    out_stride_3,
    BLOCK_CHANNEL: tl.constexpr,
):
    # Each program handles one (batch, height, width) position
    pid = tl.program_id(0)
    
    # Decode pid to get batch, height, width indices
    w = pid % width
    pid_temp = pid // width
    h = pid_temp % height
    b = pid_temp // height
    
    # Find minimum across all channels for this position
    channel_offsets = tl.arange(0, BLOCK_CHANNEL)
    mask = channel_offsets < channels
    
    # Calculate the pointer for each channel
    ptr = x_ptr + b * x_stride_0 + channel_offsets * x_stride_1 + h * x_stride_2 + w * x_stride_3
    
    # Load values
    vals = tl.load(ptr, mask=mask, other=1e10)
    
    # Find minimum across channels
    min_val = tl.min(vals, axis=0)
    
    # Apply tanh twice (fused operation)
    tanh1 = tl.libdevice.tanh(min_val)
    tanh2 = tl.libdevice.tanh(tanh1)
    
    # Store result (output has only 1 channel, so channel index is 0)
    out_ptr_offset = b * out_stride_0 + h * out_stride_2 + w * out_stride_3
    tl.store(out_ptr + out_ptr_offset, tanh2)


def min_tanh_fused(x: torch.Tensor):
    """
    Fused kernel that performs min across channel dimension followed by two tanh operations.
    Input: [batch, channels, height, width]
    Output: [batch, 1, height, width]
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    # Prepare output tensor
    out = torch.empty((batch_size, 1, height, width), dtype=x.dtype, device=x.device)
    
    # Number of positions (batch * height * width)
    n_positions = batch_size * height * width
    
    # Block size for channel dimension
    BLOCK_CHANNEL = triton.next_power_of_2(channels)
    
    # Launch the Triton kernel
    grid = (n_positions,)
    min_tanh_fused_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        BLOCK_CHANNEL=BLOCK_CHANNEL,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    Uses fused Triton kernel for min + tanh + tanh operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = min_tanh_fused(x)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 25
```
