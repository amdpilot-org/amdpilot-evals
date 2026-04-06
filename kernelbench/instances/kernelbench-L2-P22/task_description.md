# KernelBench Level 2 Problem 22: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish.py

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
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.matmul(x)
        x = x * self.scale_factor
        x = x + x
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * torch.nn.functional.mish(x)  # Mish activation
        return x

batch_size = 1024
input_size = 8192
hidden_size = 8192
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]
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

A previous single-shot attempt failed. Error: `at 15:15:
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_scale_add_clamp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale_factor,
    clamp_min,
    clamp_max,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Original: x = x * scale_factor; x = x + x
    # Equivalent to: x = x * (scale_factor + 1.0)
    out = x * (scale_factor + 1.0)
    
    # Clamp
    out = tl.maximum(out, clamp_min)
    out = tl.minimum(out, clamp_max)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_size
    
    # Online logsumexp for numerical stability
    # First pass: find maximum value in the row
    max_val = -1e10
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + hidden_size
        x = tl.load(x_ptr + offsets, mask=mask, other=-1e10)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute sum of exp(x - max)
    sum_exp = 0.0
    for block_start in range(0, hidden_size, BLOCK_SIZE):
        offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + hidden_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        exp_val = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_val, axis=0)
    
    # Compute logsumexp
    lse = max_val + tl.log(sum_exp + 1e-10)
    
    # Store result (shape: batch_size, 1)
    tl.store(out_ptr + row_idx, lse)


@triton.jit
def mish_mult_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Mish activation: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    softplus = tl.math.log(1.0 + tl.exp(x))
    mish = x * tl.math.tanh(softplus)
    
    # Original: x * mish(x)
    out = x * mish
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_scale_add_clamp(x: torch.Tensor, scale_factor: float, clamp_min: float, clamp_max: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_scale_add_clamp_kernel[grid](
        x, out, n_elements,
        scale_factor, clamp_min, clamp_max,
        BLOCK_SIZE=BLOCK_SIZE
 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 22
```
