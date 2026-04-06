# KernelBench Level 2 Problem 53: 53_Gemm_Scaling_Hardtanh_GELU.py

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
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scaling_factor
        x = self.hardtanh(x)
        x = self.gelu(x)
        return x

batch_size = 2048
in_features = 8192
out_features = 8192
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]
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

A previous single-shot attempt failed. Error: `at 32:17:
    x = x * scaling_factor

    # Apply HardTanh (clamp)
    x = tl.maximum(x, hardtanh_min)
    x = tl.minimum(x, hardtanh_max)

    # Apply GELU activation
    # GELU approximation: 0.5...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_scale_hardtanh_gelu_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    scaling_factor,  # Scaling factor
    hardtanh_min,  # HardTanh minimum
    hardtanh_max,  # HardTanh maximum
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle boundary cases
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply scaling
    x = x * scaling_factor
    
    # Apply HardTanh (clamp)
    x = tl.maximum(x, hardtanh_min)
    x = tl.minimum(x, hardtanh_max)
    
    # Apply GELU activation
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_inner = tl.math.tanh(inner)
    gelu_out = 0.5 * x * (1.0 + tanh_inner)
    
    # Store output
    tl.store(out_ptr + offsets, gelu_out, mask=mask)


def fused_scale_hardtanh_gelu(x: torch.Tensor, scaling_factor: float, 
                               hardtanh_min: float, hardtanh_max: float):
    """
    Fused kernel for scaling, HardTanh, and GELU activation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    fused_scale_hardtanh_gelu_kernel[grid](
        x, out, n_elements, 
        scaling_factor, hardtanh_min, hardtanh_max,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a GEMM, scaling, hardtanh, and GELU activation.
    Uses fused Triton kernel for scaling + hardtanh + gelu.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.gemm(x)
        x = fused_scale_hardtanh_gelu(x, self.scaling_factor, 
                                       self.hardtanh_min, self.hardtanh_max)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 53
```
