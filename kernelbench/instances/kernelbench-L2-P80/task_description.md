# KernelBench Level 2 Problem 80: 80_Gemm_Max_Subtract_GELU.py

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
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.gemm(x)
        x = torch.max(x, dim=self.max_dim, keepdim=True).values
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
max_dim = 1

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, max_dim]
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

A previous single-shot attempt failed. Error: `at 44:45:
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

        # Apply max operation
        x_max = tl.maximum(x, max_val)

        # Subtract mean
        x_centered = ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def fused_max_mean_gelu_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    n_features,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x
    out_row_start = row_idx * stride_out
    
    # First pass: compute max and sum for mean
    max_val = tl.zeros([1], dtype=tl.float32) - 1e30
    sum_val = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(0, n_features, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-1e30)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))
        x_for_sum = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x_for_sum, axis=0)
    
    mean_val = sum_val / n_features
    
    # Second pass: apply max, subtract mean, apply GELU
    sqrt_2_over_pi = 0.7978845608028654
    for block_start in range(0, n_features, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
        
        # Apply max operation
        x_max = tl.maximum(x, max_val)
        
        # Subtract mean
        x_centered = x_max - mean_val
        
        # Apply GELU approximation
        x_cubed = x_centered * x_centered * x_centered
        tanh_arg = sqrt_2_over_pi * (x_centered + 0.044715 * x_cubed)
        gelu_out = 0.5 * x_centered * (1.0 + tl.math.tanh(tanh_arg))
        
        tl.store(out_ptr + out_row_start + offsets, gelu_out, mask=mask)


def fused_max_mean_gelu(x: torch.Tensor):
    """
    Fused kernel for: max along dim=1, subtract mean along dim=1, apply GELU
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, n_features = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_features)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = (batch_size,)
    
    fused_max_mean_gelu_kernel[grid](
        x,
        out,
        batch_size,
        n_features,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a GEMM, followed by fused max, mean subtraction, and GELU.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        x = self.gemm(x)
        x = fused_max_mean_gelu(x)
        ret
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 80
```
