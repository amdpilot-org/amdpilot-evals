# KernelBench Level 2 Problem 34: 34_ConvTranspose3d_LayerNorm_GELU_Scaling.py

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
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        x = self.layer_norm(x)
        x = torch.nn.functional.gelu(x)
        x = x * self.scaling_factor
        return x

batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
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

A previous single-shot attempt failed. Error: `name 'fused_layernorm_gelu_scale_kernel' is not defined`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def layernorm_gelu_scale_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    stride_n,
    stride_d,
    stride_h,
    stride_w,
    stride_c,
    n_elements,
    n_channels,
    eps,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for LayerNorm + GELU + Scaling.
    Each program handles one row (one spatial position across all channels for one sample).
    """
    row_idx = tl.program_id(0)
    
    # Compute mean
    mean = 0.0
    for i in range(0, n_channels, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_channels
        x_ptr_row = x_ptr + row_idx * stride_n + offsets * stride_c
        x = tl.load(x_ptr_row, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / n_channels
    
    # Compute variance
    var = 0.0
    for i in range(0, n_channels, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_channels
        x_ptr_row = x_ptr + row_idx * stride_n + offsets * stride_c
        x = tl.load(x_ptr_row, mask=mask, other=0.0)
        var += tl.sum((x - mean) * (x - mean), axis=0)
    var = var / n_channels
    
    # Normalize, apply GELU, and scale
    for i in range(0, n_channels, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_channels
        x_ptr_row = x_ptr + row_idx * stride_n + offsets * stride_c
        weight_ptr_row = weight_ptr + offsets
        bias_ptr_row = bias_ptr + offsets
        out_ptr_row = out_ptr + row_idx * stride_n + offsets * stride_c
        
        x = tl.load(x_ptr_row, mask=mask, other=0.0)
        weight = tl.load(weight_ptr_row, mask=mask, other=0.0)
        bias = tl.load(bias_ptr_row, mask=mask, other=0.0)
        
        # LayerNorm
        x_norm = (x - mean) / tl.sqrt(var + eps)
        x_norm = x_norm * weight + bias
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_gelu = 0.5 * x_norm * (1.0 + tl.math.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
        
        # Scaling
        out = x_gelu * scaling_factor
        
        tl.store(out_ptr_row, out, mask=mask)


def fused_layernorm_gelu_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    scaling_factor: float,
):
    """
    Fused LayerNorm + GELU + Scaling using Triton.
    x: (N, C, D, H, W) - LayerNorm applied over C dimension
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    N, C, D, H, W = x.shape
    n_rows = N * D * H * W
    n_channels = C
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_channels)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (n_rows,)
    
    fused_layernorm_gelu
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 34
```
