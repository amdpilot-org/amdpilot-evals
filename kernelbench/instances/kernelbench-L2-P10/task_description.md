# KernelBench Level 2 Problem 10: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh.py

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
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = torch.tanh(x)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]
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

A previous single-shot attempt failed. Error: `at 77:19:
            valid_pos = (ih < H) & (iw < W) & valid_spatial
            offset = pid_b * stride_b + c_offs + ih * stride_h + iw * stride_w
            vals = tl.load(x_ptr + offset, mask=...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_activ_reduce_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    pool_kernel: tl.constexpr,
    pool_stride: tl.constexpr,
    hardtanh_min: tl.constexpr,
    hardtanh_max: tl.constexpr,
    BLOCK_C: tl.constexpr,
    MAX_SPATIAL: tl.constexpr,
):
    pid_b = tl.program_id(0)
    
    for c_start in range(0, stride_c, BLOCK_C * stride_c):
        c_offs = c_start + tl.arange(0, BLOCK_C) * stride_c
        c_mask = c_start + tl.arange(0, BLOCK_C) < stride_c
        
        sum_vals = tl.zeros([BLOCK_C], dtype=tl.float32)
        
        for spatial_idx in range(MAX_SPATIAL):
            oh = spatial_idx // OW
            ow = spatial_idx % OW
            
            valid_spatial = spatial_idx < OH * OW
            
            ih_start = oh * pool_stride
            iw_start = ow * pool_stride
            
            max_vals = tl.full([BLOCK_C], -1e10, dtype=tl.float32)
            
            # Unroll 2x2 pooling window
            # Position (0, 0)
            ih = ih_start
            iw = iw_start
            valid_pos = (ih < H) & (iw < W) & valid_spatial
            offset = pid_b * stride_b + c_offs + ih * stride_h + iw * stride_w
            vals = tl.load(x_ptr + offset, mask=c_mask & valid_pos, other=-1e10)
            max_vals = tl.maximum(max_vals, vals)
            
            # Position (0, 1)
            ih = ih_start
            iw = iw_start + 1
            valid_pos = (ih < H) & (iw < W) & valid_spatial
            offset = pid_b * stride_b + c_offs + ih * stride_h + iw * stride_w
            vals = tl.load(x_ptr + offset, mask=c_mask & valid_pos, other=-1e10)
            max_vals = tl.maximum(max_vals, vals)
            
            # Position (1, 0)
            ih = ih_start + 1
            iw = iw_start
            valid_pos = (ih < H) & (iw < W) & valid_spatial
            offset = pid_b * stride_b + c_offs + ih * stride_h + iw * stride_w
            vals = tl.load(x_ptr + offset, mask=c_mask & valid_pos, other=-1e10)
            max_vals = tl.maximum(max_vals, vals)
            
            # Position (1, 1)
            ih = ih_start + 1
            iw = iw_start + 1
            valid_pos = (ih < H) & (iw < W) & valid_spatial
            offset = pid_b * stride_b + c_offs + ih * stride_h + iw * stride_w
            vals = tl.load(x_ptr + offset, mask=c_mask & valid_pos, other=-1e10)
            max_vals = tl.maximum(max_vals, vals)
            
            # Apply HardTanh
            clipped = tl.maximum(hardtanh_min, tl.minimum(hardtanh_max, max_vals))
            sum_vals += clipped * valid_spatial
        
        # Compute mean and apply Tanh
        mean_vals = sum_vals / OH / OW
        out_vals = tl.libdevice.tanh(mean_vals)
        
        # Store
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 10
```
