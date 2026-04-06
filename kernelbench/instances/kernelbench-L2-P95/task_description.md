# KernelBench Level 2 Problem 95: 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh.py

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
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        x = self.matmul(x)
        x = x + self.add_value
        x = torch.sigmoid(x) * x # Swish
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x) # GELU
        x = torch.nn.functional.hardtanh(x, min_val=-1, max_val=1) # Hardtanh
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
add_value_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]
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

A previous single-shot attempt failed. Error: `at 35:8:
    # Load bias values
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)

    # Add bias
    x = x + bias

    # Apply Swish: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl....`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_activation_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Calculate the starting offset for this row
    row_start = row_idx * stride_x
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input values
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # Load bias values
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # Add bias
    x = x + bias
    
    # Apply Swish: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    x = x * sigmoid_x
    
    # Apply Tanh
    x = tl.libdevice.tanh(x)
    
    # Apply GELU (approximation)
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_inner = tl.libdevice.tanh(inner)
    x = 0.5 * x * (1.0 + tanh_inner)
    
    # Apply Hardtanh (clamp to [-1, 1])
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)
    
    # Store output
    out_start = row_idx * stride_out
    tl.store(out_ptr + out_start + col_offsets, x, mask=mask)


def fused_activation(x: torch.Tensor, bias: torch.Tensor):
    """
    Fused kernel that applies: add bias + Swish + Tanh + GELU + Hardtanh
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Block size (tunable parameter)
    BLOCK_SIZE = 1024
    
    # Determine the number of blocks needed (one block per row)
    grid = (n_rows,)
    
    # Get strides
    stride_x = x.stride(0)
    stride_out = out.stride(0)
    
    # Launch the Triton kernel
    fused_activation_kernel[grid](
        x, bias, out,
        n_rows, n_cols,
        stride_x, stride_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication followed by 
    fused activation functions (add + Swish + Tanh + GELU + Hardtanh) 
    using a custom Triton kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        # Matrix multiplication (keep PyTorch's optimized implementation)
        x = self.matmul(x)
        
        # Fused activation: add bias + Swish + Tanh + GELU + Hardtanh
        x = fused_activation(x, self.add_value)
  
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 95
```
