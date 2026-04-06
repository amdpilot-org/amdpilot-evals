# KernelBench Level 1 Problem 52: 52_Argmin_over_a_dimension.py

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
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        return torch.argmin(x, dim=self.dim)

batch_size = 128
dim1 = 4096
dim2 = 4095
dim = 1

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

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

A previous single-shot attempt failed. Error: `at 19:20:
):
    """
    Triton kernel for argmin operation.
    Each program computes argmin for one row.
    """
    pid = tl.program_id(0)

    # Base offset for this row
    base_offset = pid *...`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,
    out_ptr,
    stride_row,
    stride_col,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for argmin operation.
    Each program computes argmin for one row.
    """
    pid = tl.program_id(0)
    
    # Base offset for this row
    base_offset = pid * stride_row
    
    # Initialize accumulators with infinity
    acc_min_value = tl.float32(float('inf'))
    acc_min_index = 0
    
    # Process the row in blocks
    for block_start in range(0, dim_size, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Load values from input tensor
        load_offsets = base_offset + offsets * stride_col
        values = tl.load(x_ptr + load_offsets, mask=mask, other=float('inf'))
        
        # Find minimum value in this block
        block_min_value = tl.min(values, axis=0)
        
        # Find index of minimum in this block
        is_min = values == block_min_value
        block_indices = offsets
        masked_indices = tl.where(is_min & mask, block_indices, tl.int32(dim_size))
        block_min_index = tl.min(masked_indices, axis=0)
        
        # Update accumulators
        should_update = block_min_value < acc_min_value
        same_value = block_min_value == acc_min_value
        better_index = block_min_index < acc_min_index
        
        acc_min_value = tl.where(should_update, block_min_value, acc_min_value)
        update_index = should_update | (same_value & better_index)
        acc_min_index = tl.where(update_index, block_min_index, acc_min_index)
    
    # Store result
    tl.store(out_ptr + pid, acc_min_index)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-based argmin implementation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    # Normalize dim to be positive
    if dim < 0:
        dim = x.dim() + dim
    
    dim_size = x.shape[dim]
    
    # Calculate other_size (number of rows after transpose)
    other_size = 1
    for i in range(x.dim()):
        if i != dim:
            other_size *= x.shape[i]
    
    # Reshape to 2D for simpler kernel
    # Move reduction dim to last position
    x_2d = x.transpose(dim, -1).reshape(other_size, dim_size).contiguous()
    
    # Output tensor
    out = torch.empty(other_size, dtype=torch.int64, device=x.device)
    
    # Tunable block size
    BLOCK_SIZE = 1024
    
    # Grid size - one program per output element
    grid = (other_size,)
    
    # Launch kernel
    argmin_kernel[grid](
        x_2d,
        out,
        x_2d.stride(0),
        x_2d.stride(1),
        dim_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to match expected shape
    out_shape = list(x.shape)
    out_shape.pop(dim)
    out = out.view(out_shape)
    
    return out


class ModelNew(nn.Modul
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 52
```
