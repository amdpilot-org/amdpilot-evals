# KernelBench Level 2 Problem 43: 43_Conv3d_Max_LogSumExp_ReLU.py

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
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.relu(x)
        return x

batch_size = 4
in_channels = 32
out_channels = 64
depth, height, width = 32, 128, 128
kernel_size = 3
stride = 1
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
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

A previous single-shot attempt failed. Error: `Evaluation timed out`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_logsumexp_relu_kernel(
    x_ptr,
    out_ptr,
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
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel for LogSumExp along channel dimension + ReLU activation.
    Each program handles one spatial location (b, d, h, w) across all channels.
    """
    # Get the spatial location this program is responsible for
    pid = tl.program_id(0)
    
    # Decode the batch and spatial indices from pid
    total_spatial = depth * height * width
    b_idx = pid // total_spatial
    spatial_idx = pid % total_spatial
    d_idx = spatial_idx // (height * width)
    hw_idx = spatial_idx % (height * width)
    h_idx = hw_idx // width
    w_idx = hw_idx % width
    
    # Compute base offset for this spatial location
    base_offset = (b_idx * stride_b + d_idx * stride_d + 
                   h_idx * stride_h + w_idx * stride_w)
    
    # First pass: find maximum value across channels for numerical stability
    max_val = -float('inf')
    for c_start in range(0, channels, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask = c_offsets < channels
        offsets = base_offset + c_offsets * stride_c
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_vals, axis=0))
    
    # Second pass: compute sum of exp(x - max_val)
    sum_exp = 0.0
    for c_start in range(0, channels, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask = c_offsets < channels
        offsets = base_offset + c_offsets * stride_c
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        exp_vals = tl.exp(x_vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)
    
    # Compute logsumexp
    logsumexp_val = max_val + tl.log(sum_exp)
    
    # Apply ReLU
    output_val = tl.maximum(logsumexp_val, 0.0)
    
    # Store result (output has only 1 channel)
    out_offset = base_offset  # Only 1 channel in output, stride_c effectively 0 for output
    tl.store(out_ptr + out_offset, output_val)


def fused_logsumexp_relu(x: torch.Tensor):
    """
    Fused LogSumExp along dim=1 + ReLU using Triton.
    Input: (B, C, D, H, W)
    Output: (B, 1, D, H, W)
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    
    batch_size, channels, depth, height, width = x.shape
    
    # Prepare output tensor with 1 channel
    out = torch.empty((batch_size, 1, depth, height, width), dtype=x.dtype, device=x.device)
    
    # Calculate strides
    stride_b = x.stride(0)
    stride_c = x.stride(1)
    stride_d = x.stride(2)
    stride_h = x.stride(3)
    stride_w = x.stride(4)
    
    # Total number of spatial locations (each program handles one)
    total_programs = batch_size * depth 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 43
```
