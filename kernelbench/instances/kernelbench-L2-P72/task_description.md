# KernelBench Level 2 Problem 72: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool.py

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
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        return x


batch_size = 64
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
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

A previous single-shot attempt failed. Error: `Cannot call @triton.jit'd outside of the scope of a kernel`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_avgpool_kernel(
    x_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    out_ptr,
    batch_size,
    channels,
    in_depth,
    in_height,
    in_width,
    out_depth,
    out_height,
    out_width,
    pool_kernel: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(out_depth * out_height * out_width, BLOCK_D * BLOCK_H * BLOCK_W)
    batch_id = pid // num_blocks
    spatial_id = pid % num_blocks
    
    d_start = (spatial_id // (BLOCK_H * BLOCK_W)) * BLOCK_D
    h_start = ((spatial_id // BLOCK_W) % BLOCK_H) * BLOCK_H
    w_start = (spatial_id % BLOCK_W) * BLOCK_W
    
    for c_off in range(0, channels, BLOCK_C):
        c_idx = c_off + tl.arange(0, BLOCK_C)
        c_mask = c_idx < channels
        
        for d_off in range(BLOCK_D):
            for h_off in range(BLOCK_H):
                for w_off in range(BLOCK_W):
                    out_d = d_start + d_off
                    out_h = h_start + h_off
                    out_w = w_start + w_off
                    
                    out_mask = (out_d < out_depth) & (out_h < out_height) & (out_w < out_width)
                    
                    if tl.sum(out_mask) > 0:
                        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
                        count = 0
                        
                        for pd in range(pool_kernel):
                            for ph in range(pool_kernel):
                                for pw in range(pool_kernel):
                                    in_d = out_d * pool_kernel + pd
                                    in_h = out_h * pool_kernel + ph
                                    in_w = out_w * pool_kernel + pw
                                    
                                    in_mask = (in_d < in_depth) & (in_h < in_height) & (in_w < in_width)
                                    
                                    if tl.sum(in_mask) > 0:
                                        in_idx = (((batch_id * channels + c_idx) * in_depth + in_d) * in_height + in_h) * in_width + in_w
                                        x_val = tl.load(x_ptr + in_idx, mask=c_mask & in_mask, other=0.0)
                                        acc += x_val
                                        count += 1
                        
                        acc = acc / count
                        
                        bn_weight = tl.load(bn_weight_ptr + c_idx, mask=c_mask, other=1.0)
                        bn_bias = tl.load(bn_bias_ptr + c_idx, mask=c_mask, other=0.0)
                        bn_mean = tl.load(bn_mean_ptr + c_idx, mask=c_mask, other=0.0)
                        bn_var = tl.load(bn_var_ptr + c_idx, mask=c_mask, other=1.0)
                        

```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 72
```
