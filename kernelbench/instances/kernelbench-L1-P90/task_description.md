# KernelBench Level 1 Problem 90: 90_cumprod.py

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
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        return torch.cumprod(x, dim=self.dim)

# Define input dimensions and parameters
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]

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
def cumprod_kernel(
    x_ptr,
    out_ptr,
    stride_batch,
    stride_dim,
    n_elements_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for cumulative product along a dimension.
    Each program handles one row (batch element) and computes cumprod along the dimension.
    Uses a parallel scan approach within the block.
    """
    batch_idx = tl.program_id(0)
    batch_start = batch_idx * stride_batch

    # Load input values for this row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements_dim
    x = tl.load(x_ptr + batch_start + offsets * stride_dim, mask=mask, other=1.0)

    # Parallel scan for cumulative product using Blelloch algorithm
    # Up-sweep phase (reduce)
    temp = x
    for k in range(1, tl.constexpr(BLOCK_SIZE).value):
        if (1 << k) <= BLOCK_SIZE:
            shift = 1 << (k - 1)
            temp_prev = tl.where(offsets >= shift, temp, 1.0)
            temp_curr = tl.where(offsets >= shift, tl.load(x_ptr + batch_start + (offsets - shift) * stride_dim, mask=(offsets - shift) < n_elements_dim, other=1.0), 1.0)
            temp = tl.where(offsets >= shift, temp_prev * temp_curr, temp)

    # Down-sweep phase (scan)
    cumprod = tl.where(offsets == 0, 1.0, x)
    for k in range(tl.constexpr(BLOCK_SIZE).value.bit_length() - 1, -1, -1):
        if (1 << k) <= BLOCK_SIZE:
            shift = 1 << k
            temp_prev = tl.where(offsets >= shift, cumprod, 1.0)
            temp_curr = tl.where(offsets >= shift, tl.load(x_ptr + batch_start + (offsets - shift) * stride_dim, mask=(offsets - shift) < n_elements_dim, other=1.0), 1.0)
            cumprod = tl.where(offsets >= shift, temp_prev * temp_curr, cumprod)

    # Final cumulative product
    cumprod = cumprod * x

    # Store output
    tl.store(out_ptr + batch_start + offsets * stride_dim, cumprod, mask=mask)


def triton_cumprod(x: torch.Tensor, dim: int):
    """
    Wrapper function for Triton cumulative product kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Normalize negative dimension
    if dim < 0:
        dim = x.dim() + dim

    # Transpose so that the cumprod dimension is the last dimension
    if dim != x.dim() - 1:
        x = x.transpose(dim, -1)
        transpose_back = True
    else:
        transpose_back = False

    # Output tensor
    out = torch.empty_like(x)

    # Get dimensions
    batch_size = x.numel() // x.shape[-1]
    n_elements_dim = x.shape[-1]

    # Determine block size (must be power of 2 and >= n_elements_dim for simplicity)
    BLOCK_SIZE = triton.next_power_of_2(n_elements_dim)

    # Grid: one block per batch element
    grid = (batch_size,)

    # Strides
    stride_batch = x.stride(0)
    stride_dim = x.stride(-1)

    # Launch kernel
    cumprod_kernel[grid](
        x,
        out,
        stride_batch,
        stride_dim,
        n_elements_dim,
     
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 90
```
