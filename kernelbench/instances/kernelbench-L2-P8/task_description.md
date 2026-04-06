# KernelBench Level 2 Problem 8: 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py

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
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]
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
def fused_post_conv_kernel(
    conv_out_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    divisor,
    pool_d,
    pool_h,
    pool_w,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    BLOCK_CHANNEL: tl.constexpr,
    POOLED_D: tl.constexpr,
    POOLED_H: tl.constexpr,
    POOLED_W: tl.constexpr,
):
    """
    Fused kernel for post-convolution operations:
    1. Division by divisor
    2. MaxPool3d
    3. GlobalAvgPool3d
    4. Bias addition
    5. Sum across channels
    
    Input shape: (N, C, D, H, W)
    Output shape: (N,)
    """
    batch_idx = tl.program_id(0)
    
    # Channel offsets
    ch_offsets = tl.arange(0, BLOCK_CHANNEL)
    ch_mask = ch_offsets < channels
    
    # Load bias for all channels
    bias = tl.load(bias_ptr + ch_offsets, mask=ch_mask, other=0.0)
    
    # Total spatial positions after pooling
    total_spatial = POOLED_D * POOLED_H * POOLED_W
    
    # Accumulator for channel values
    channel_acc = tl.zeros([BLOCK_CHANNEL], dtype=tl.float32)
    
    # Iterate over pooled spatial dimensions
    for pd in tl.static_range(POOLED_D):
        for ph in tl.static_range(POOLED_H):
            for pw in tl.static_range(POOLED_W):
                # Find max in pool window for each channel
                max_vals = tl.zeros([BLOCK_CHANNEL], dtype=tl.float32)
                
                # Pool window iteration (pool_size is 2x2x2)
                for kd in tl.static_range(2):
                    for kh in tl.static_range(2):
                        for kw in tl.static_range(2):
                            orig_d = pd * pool_d + kd
                            orig_h = ph * pool_h + kh
                            orig_w = pw * pool_w + kw
                            
                            # Bounds check
                            if orig_d < depth and orig_h < height and orig_w < width:
                                # Compute indices for each channel
                                base_idx = batch_idx * stride_b + ch_offsets * stride_c
                                idx = base_idx + orig_d * stride_d + orig_h * stride_h + orig_w * stride_w
                                
                                # Load values, apply division, and find max
                                vals = tl.load(conv_out_ptr + idx, mask=ch_mask, other=0.0)
                                vals = vals / divisor
                                max_vals = tl.maximum(max_vals, vals)
                
                # Accumulate for global average
                channel_acc += max_vals
    
    # Global average pool: divide by total spatial positions
    channel_acc = channel_acc / total_spatial
    
    # Add bias
    channel_acc = channel_acc + bias
    
    # Sum across channels
    result = tl.sum(channel_acc, axis=0)
    
    # Store output
    tl.store(out_ptr
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 8
```
