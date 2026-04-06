# KernelBench Level 2 Problem 36: 36_ConvTranspose2d_Min_Sum_GELU_Add.py

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
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.min(x, dim=1, keepdim=True)[0]  # Minimum operation along channel dimension
        x = torch.sum(x, dim=2, keepdim=True)  # Sum operation along height dimension
        x = torch.nn.functional.gelu(x)  # GELU activation
        x = x + self.bias
        return x

batch_size = 16
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
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

A previous single-shot attempt failed. Error: `at 35:32:
    for h in range(height):
        min_val = 1e10
        for c in range(channels):
            offset = batch_idx * stride_b + c * stride_c + h * stride_h + width_idx * stride_w
       ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_min_sum_gelu_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    stride_b,
    stride_c,
    stride_h,
    stride_w,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * width:
        return
    
    batch_idx = pid // width
    width_idx = pid % width
    
    total_sum = 0.0
    
    for h in range(height):
        min_val = 1e10
        for c in range(channels):
            offset = batch_idx * stride_b + c * stride_c + h * stride_h + width_idx * stride_w
            val = tl.load(x_ptr + offset)
            min_val = tl.minimum(min_val, val)
        total_sum = total_sum + min_val
    
    x = total_sum
    x3 = x * x * x
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x3)
    gelu_val = 0.5 * x * (1.0 + tl.libdevice.tanh(tanh_arg))
    
    bias_val = tl.load(bias_ptr)
    out_val = gelu_val + bias_val
    
    tl.store(out_ptr + pid, out_val)


def fused_min_sum_gelu_bias(x, bias):
    assert x.is_cuda, "Input must be on CUDA"
    assert bias.is_cuda, "Bias must be on CUDA"
    
    x = x.contiguous()
    bias = bias.contiguous()
    
    batch_size, channels, height, width = x.shape
    
    out = torch.empty(batch_size, 1, 1, width, device=x.device, dtype=x.dtype)
    
    stride_b = x.stride(0)
    stride_c = x.stride(1)
    stride_h = x.stride(2)
    stride_w = x.stride(3)
    
    grid = (batch_size * width,)
    
    fused_min_sum_gelu_bias_kernel[grid](
        x, bias, out,
        batch_size, channels, height, width,
        stride_b, stride_c, stride_h, stride_w,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_min_sum_gelu_bias(x, self.bias)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 36
```
