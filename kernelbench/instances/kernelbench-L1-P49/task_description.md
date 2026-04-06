# KernelBench Level 1 Problem 49: 49_Max_reduction_over_a_dimension.py

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
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        return torch.max(x, dim=self.dim)[0]

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension
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

A previous single-shot attempt failed. Error: `object of type 'int' has no len()`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_dim0_kernel(
    x_ptr,
    out_ptr,
    stride_x0, stride_x1, stride_x2,
    stride_out0, stride_out1,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Output shape: (dim1, dim2), each program handles one output element
    pid = tl.program_id(0)
    out_idx1 = pid // dim2
    out_idx2 = pid % dim2

    max_val = -1e30  # Initialize with a very small number for FP32

    for i in range(0, dim0, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim0
        x_idx = offsets * stride_x0 + out_idx1 * stride_x1 + out_idx2 * stride_x2
        x = tl.load(x_ptr + x_idx, mask=mask, other=-1e30)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    out_idx = out_idx1 * stride_out0 + out_idx2 * stride_out1
    tl.store(out_ptr + out_idx, max_val)


@triton.jit
def max_reduce_dim1_kernel(
    x_ptr,
    out_ptr,
    stride_x0, stride_x1, stride_x2,
    stride_out0, stride_out1,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Output shape: (dim0, dim2), each program handles one output element
    pid = tl.program_id(0)
    out_idx0 = pid // dim2
    out_idx2 = pid % dim2

    max_val = -1e30

    for i in range(0, dim1, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim1
        x_idx = out_idx0 * stride_x0 + offsets * stride_x1 + out_idx2 * stride_x2
        x = tl.load(x_ptr + x_idx, mask=mask, other=-1e30)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    out_idx = out_idx0 * stride_out0 + out_idx2 * stride_out1
    tl.store(out_ptr + out_idx, max_val)


@triton.jit
def max_reduce_dim2_kernel(
    x_ptr,
    out_ptr,
    stride_x0, stride_x1, stride_x2,
    stride_out0, stride_out1,
    dim0: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Output shape: (dim0, dim1), each program handles one output element
    pid = tl.program_id(0)
    out_idx0 = pid // dim1
    out_idx1 = pid % dim1

    max_val = -1e30

    for i in range(0, dim2, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim2
        x_idx = out_idx0 * stride_x0 + out_idx1 * stride_x1 + offsets * stride_x2
        x = tl.load(x_ptr + x_idx, mask=mask, other=-1e30)
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    out_idx = out_idx0 * stride_out0 + out_idx1 * stride_out1
    tl.store(out_ptr + out_idx, max_val)


def triton_max_reduce(x: torch.Tensor, dim: int):
    """
    Triton-based max reduction over a specified dimension.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    ndim = x.ndim
    assert ndim == 3, "This kernel supports 3D tensors only."

  
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 49
```
