# KernelBench Level 2 Problem 50: 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling.py

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
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x * self.scale1
        x = self.avg_pool(x)
        x = x + self.bias
        x = x * self.scale2
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]
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

A previous single-shot attempt failed. Error: `at 144:14:
    input_idx = out_n * stride_xn + out_c * stride_xc + in_d * stride_xd + in_h * stride_xh + in_w * stride_xw
    acc += tl.load(x_ptr + input_idx, mask=valid, other=0.0)
    count += v...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_scale_bias_kernel(
    x_ptr,
    out_ptr,
    bias_ptr,
    batch_size,
    channels,
    in_depth,
    in_height,
    in_width,
    out_depth,
    out_height,
    out_width,
    scale1,
    scale2,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Decode output indices from linear index
    tmp = pid
    out_w = tmp % out_width
    tmp = tmp // out_width
    out_h = tmp % out_height
    tmp = tmp // out_height
    out_d = tmp % out_depth
    tmp = tmp // out_depth
    out_c = tmp % channels
    out_n = tmp // channels
    
    # Input coordinates for 2x2x2 pooling
    in_d_start = out_d * 2
    in_h_start = out_h * 2
    in_w_start = out_w * 2
    
    # Unrolled 2x2x2 average pooling (8 positions)
    acc = 0.0
    count = 0.0
    
    # Position (0,0,0)
    in_d = in_d_start
    in_h = in_h_start
    in_w = in_w_start
    in_d_valid = in_d < in_depth
    in_h_valid = in_h < in_height
    in_w_valid = in_w < in_width
    valid = in_d_valid & in_h_valid & in_w_valid
    input_idx = out_n * stride_xn + out_c * stride_xc + in_d * stride_xd + in_h * stride_xh + in_w * stride_xw
    acc += tl.load(x_ptr + input_idx, mask=valid, other=0.0)
    count += valid.to(tl.float32)
    
    # Position (0,0,1)
    in_d = in_d_start
    in_h = in_h_start
    in_w = in_w_start + 1
    in_d_valid = in_d < in_depth
    in_h_valid = in_h < in_height
    in_w_valid = in_w < in_width
    valid = in_d_valid & in_h_valid & in_w_valid
    input_idx = out_n * stride_xn + out_c * stride_xc + in_d * stride_xd + in_h * stride_xh + in_w * stride_xw
    acc += tl.load(x_ptr + input_idx, mask=valid, other=0.0)
    count += valid.to(tl.float32)
    
    # Position (0,1,0)
    in_d = in_d_start
    in_h = in_h_start + 1
    in_w = in_w_start
    in_d_valid = in_d < in_depth
    in_h_valid = in_h < in_height
    in_w_valid = in_w < in_width
    valid = in_d_valid & in_h_valid & in_w_valid
    input_idx = out_n * stride_xn + out_c * stride_xc + in_d * stride_xd + in_h * stride_xh + in_w * stride_xw
    acc += tl.load(x_ptr + input_idx, mask=valid, other=0.0)
    count += valid.to(tl.float32)
    
    # Position (0,1,1)
    in_d = in_d_start
    in_h = in_h_start + 1
    in_w = in_w_start + 1
    in_d_valid = in_d < in_depth
    in_h_valid = in_h < in_height
    in_w_valid = in_w < in_width
    valid = in_d_valid & in_h_valid & in_w_valid
    input_idx = out_n * stride_xn + out_c * stride_xc + in_d * stride_xd + in_h * stride_xh + in_w * stride_xw
    acc += tl.load(x_ptr + input_idx, mask=valid, other=0.0)
    count += valid.to(tl.float32)
    
    # Position (1,0,0)
    in_d = in_d_start + 1
    in_h = in_h_start
    in_w = in_w_start
    in_d_valid = in_d < in_depth
    in_h_valid = in_h <
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 50
```
