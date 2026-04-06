# KernelBench Level 2 Problem 96: 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp.py

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
    Model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]
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
import triton
import triton.language as tl


@triton.jit
def fused_scale_gap_clamp_kernel(
    x_ptr,
    out_ptr,
    scale,
    clamp_min,
    clamp_max,
    spatial_size,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    out_stride_b,
    out_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    
    # Assume we have a way to get batch_size and channels from context
    # We'll pass these as additional parameters
    batch_size = tl.constexpr(128)  # Will be overridden
    channels = tl.constexpr(16)     # Will be overridden
    
    b = pid // channels
    c = pid % channels
    
    # Compute global average pooling over spatial dimensions
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = 0
    
    # Loop over spatial dimensions in blocks
    for block_start in range(0, spatial_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_size
        
        # Convert flat spatial index to 3D indices
        d_idx = offsets // (32 * 32)  # height * width
        hw_idx = offsets % (32 * 32)
        h_idx = hw_idx // 32
        w_idx = hw_idx % 32
        
        # Compute memory offset
        spatial_offset = d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        base_offset = b * stride_b + c * stride_c
        
        # Load values
        vals = tl.load(x_ptr + base_offset + spatial_offset, mask=mask, other=0.0)
        acc += vals
        count += tl.sum(mask.to(tl.int32))
    
    # Compute mean
    mean = tl.sum(acc) / count
    
    # Apply scale
    scaled = mean * scale
    
    # Apply clamp
    clamped = tl.maximum(tl.minimum(scaled, clamp_max), clamp_min)
    
    # Store result
    out_offset = b * out_stride_b + c * out_stride_c
    tl.store(out_ptr + out_offset, clamped)


@triton.jit
def fused_scale_gap_clamp_kernel_v2(
    x_ptr,
    out_ptr,
    scale,
    clamp_min,
    clamp_max,
    D,
    H,
    W,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    out_stride_b,
    out_stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch_size = tl.program_id(1)
    channels = tl.program_id(2)
    
    # Decode batch and channel from pid
    b = pid // channels
    c = pid % channels
    
    # Compute global average pooling over spatial dimensions
    acc = 0.0
    count = 0
    
    # Iterate over all spatial positions
    for d in range(D):
        for h in range(H):
            for w in range(W):
                # Compute memory offset
                offset = b * stride_b + c * stride_c + d * stride_d + h * stride_h + w * stride_w
                val = tl.load(x_ptr + offset)
                acc += val
                count += 1
    
    # Compute mean
    mean = acc / count
    
    # Apply scale
    scaled = mean * scale
    
 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 96
```
