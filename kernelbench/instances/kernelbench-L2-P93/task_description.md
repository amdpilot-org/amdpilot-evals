# KernelBench Level 2 Problem 93: 93_ConvTranspose2d_Add_Min_GELU_Multiply.py

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
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.add_value
        x = torch.min(x, torch.tensor(0.0, device=x.device))
        x = torch.nn.functional.gelu(x)
        x = x * self.multiply_value
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]
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

A previous single-shot attempt failed. Error: `at 30:17:
    # Add the constant value
    x = x + add_value

    # Take minimum with 0 (clamp positive values to 0)
    x = tl.minimum(x, 0.0)

    # Apply GELU activation
    # GELU approximation...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    add_value,  # Value to add
    multiply_value,  # Value to multiply
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask to handle boundary cases
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Add the constant value
    x = x + add_value

    # Take minimum with 0 (clamp positive values to 0)
    x = tl.minimum(x, 0.0)

    # Apply GELU activation
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x * x * x
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_inner = tl.libdevice.tanh(inner)
    x = 0.5 * x * (1.0 + tanh_inner)

    # Multiply by the constant value
    x = x * multiply_value

    # Store the result
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_post_conv(x: torch.Tensor, add_value: float, multiply_value: float):
    """
    Fused kernel for: (x + add_value), min(x, 0), GELU, * multiply_value
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
    fused_post_conv_kernel[grid](
        x, out, n_elements,
        add_value, multiply_value,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, then fuses
    add, min, GELU, and multiply operations into a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_conv(x, self.add_value, self.multiply_value)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 93
```
