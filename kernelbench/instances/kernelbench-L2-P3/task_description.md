# KernelBench Level 2 Problem 3: 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py

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
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x

batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]
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

A previous single-shot attempt failed. Error: `at 56:35:

    # Apply layer norm weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
        x_no...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_add_layernorm_gelu_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    sum_weight,
    N,
    C,
    D,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one spatial location (n, d, h, w) across all channels
    pid = tl.program_id(0)
    n = pid // (D * H * W)
    remainder = pid % (D * H * W)
    d = remainder // (H * W)
    remainder = remainder % (H * W)
    h = remainder // W
    w = remainder % W

    # Compute base offset for this spatial location
    base_offset = n * C * D * H * W + d * C * H * W + h * C * W + w * C

    # Load all channels for this spatial location and compute mean
    channel_offsets = tl.arange(0, BLOCK_SIZE)
    mask = channel_offsets < C
    offsets = base_offset + channel_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Add sum_weight
    x = x + sum_weight
    
    # Compute mean across channels
    mean = tl.sum(x, axis=0) / C
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / C
    
    # Normalize
    x_norm = x_centered / tl.sqrt(variance + eps)
    
    # Apply layer norm weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
        x_norm = x_norm * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm + bias
    
    # Apply GELU activation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_gelu = 0.5 * x_norm * (1.0 + tl.math.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    
    # Store output
    tl.store(out_ptr + offsets, x_gelu, mask=mask)


def fused_add_layernorm_gelu(x, sum_weight, weight=None, bias=None, eps=1e-5):
    """
    Fused kernel for Add + LayerNorm + GELU
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    
    # Ensure weight and bias are contiguous if provided
    if weight is not None:
        weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Block size should be >= C for efficiency
    BLOCK_SIZE = triton.next_power_of_2(C)
    
    # Grid: one block per spatial location
    grid = (N * D * H * W,)
    
    fused_add_layernorm_gelu_kernel[grid](
        x,
        out,
        weight,
        bias,
        sum_weight,
        N, C, D, H, W,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@triton.jit
def avgpool3d_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    kernel_d,
    kernel_h,
    kernel_w,
    stride_d,
    stride_h,
    stride_w,

```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 3
```
