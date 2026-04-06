# KernelBench Level 2 Problem 11: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm.py

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
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        return x

batch_size = 512
in_channels  = 64  
out_channels = 128  
height = width = 2048  
kernel_size  = 5
stride       = 1  
padding      = 1
groups       = 8
num_groups   = 8
height, width = 32, 32

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]
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

A previous single-shot attempt failed. Error: `at 34:18:
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)

    base_offset = batch_id * stride_bn + channel_id * stride_h
    for h in range(height):
       ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batch_norm_tanh_fused_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    stride_bn,
    stride_h,
    stride_w,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    channel_id = pid % channels
    batch_id = pid // channels
    
    mean = tl.load(mean_ptr + channel_id)
    var = tl.load(var_ptr + channel_id)
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    
    base_offset = batch_id * stride_bn + channel_id * stride_h
    for h in range(height):
        for w_block in range(0, width, BLOCK_SIZE):
            w_offsets = w_block + tl.arange(0, BLOCK_SIZE)
            mask = w_offsets < width
            offsets = base_offset + h * stride_w + w_offsets
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            normalized = (x - mean) / tl.sqrt(var + eps) * weight + bias
            out = tl.math.tanh(normalized)
            tl.store(out_ptr + offsets, out, mask=mask)


def batch_norm_tanh_fused(x, mean, var, weight, bias, eps=1e-5):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, channels, height, width = x.shape
    BLOCK_SIZE = 16
    grid = (batch_size * channels,)
    batch_norm_tanh_fused_kernel[grid](
        x, out, mean, var, weight, bias,
        x.stride(0), x.stride(1), x.stride(2),
        channels, height, width, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def max_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_size,
    stride,
    stride_bn,
    stride_ch,
    stride_h,
    stride_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // (channels * out_height * out_width)
    remainder = pid % (channels * out_height * out_width)
    channel_id = remainder // (out_height * out_width)
    remainder = remainder % (out_height * out_width)
    out_h = remainder // out_width
    out_w = remainder % out_width
    
    in_h_start = out_h * stride
    in_w_start = out_w * stride
    
    max_val = -1e10
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            if in_h < in_height and in_w < in_width:
                offset = (batch_id * stride_bn + channel_id * stride_ch +
                          in_h * stride_h + in_w * stride_w)
                val = tl.load(x_ptr + offset)
                max_val = tl.maximum(max_val, val)
    
    out_offset = (batch_id * channels * out_height * out_width +
                  channel_id * out_height * out_width +
                  out_h * out_width + out_w)
    tl.store(out_ptr + out_offset, max_val)


def max_pool2d_tr
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 11
```
