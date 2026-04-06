# KernelBench Level 2 Problem 73: 73_Conv2d_BatchNorm_Scaling.py

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
    Simple model that performs a convolution, applies Batch Normalization, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scaling_factor
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
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
def fused_bn_scale_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    bias_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    out = x * scale + bias
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_scale(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bn_scale_kernel[grid](x, out, scale, bias, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses Conv2d + BatchNorm2d + Scaling into
    Conv2d followed by a fused Triton kernel for BN and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.out_channels = out_channels
        self.scaling_factor = scaling_factor
        self.eps = 1e-5

        # Initialize BatchNorm parameters (as buffers for inference)
        self.register_buffer('bn_weight', torch.ones(out_channels))
        self.register_buffer('bn_bias', torch.zeros(out_channels))
        self.register_buffer('bn_running_mean', torch.zeros(out_channels))
        self.register_buffer('bn_running_var', torch.ones(out_channels))

        # Precompute fused scale and bias for BN + scaling
        self._compute_fused_params()

    def _compute_fused_params(self):
        # BN formula: y = (x - mean) / sqrt(var + eps) * weight + bias
        # Fused with scaling: y = x * scale + bias
        bn_scale = self.bn_weight / torch.sqrt(self.bn_running_var + self.eps)
        bn_offset = self.bn_bias - self.bn_running_mean * bn_scale

        # Apply scaling factor
        fused_scale = bn_scale * self.scaling_factor
        fused_bias = bn_offset * self.scaling_factor

        # Reshape for broadcasting: [1, out_channels, 1, 1]
        self.register_buffer('fused_scale', fused_scale.view(1, self.out_channels, 1, 1))
        self.register_buffer('fused_bias', fused_bias.view(1, self.out_channels, 1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = fused_bn_scale(x, self.fused_scale, self.fused_bias)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 73
```
