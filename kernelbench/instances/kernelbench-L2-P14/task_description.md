# KernelBench Level 2 Problem 14: 14_Gemm_Divide_Sum_Scaling.py

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
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = torch.matmul(x, self.weight.T)  # Gemm
        x = x / 2  # Divide
        x = torch.sum(x, dim=1, keepdim=True) # Sum
        x = x * self.scaling_factor  # Scaling
        return x


batch_size   = 1024  
input_size   = 8192  
hidden_size  = 8192 
scaling_factor = 1.5

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
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

A previous single-shot attempt failed. The output was incorrect.

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_matmul_sum_scale_kernel(
    x_ptr,           # Pointer to input tensor (batch_size, input_size)
    weight_sum_ptr,  # Pointer to precomputed weight sum (input_size,)
    out_ptr,         # Pointer to output tensor (batch_size, 1)
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    
    # Initialize accumulator for the sum
    acc = tl.zeros([], dtype=tl.float32)
    
    # Loop over input_size in blocks
    for block_start in range(0, input_size, BLOCK_SIZE):
        # Create offsets for this block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Mask to handle cases where input_size is not a multiple of BLOCK_SIZE
        mask = offsets < input_size
        
        # Load input values for this batch element
        x = tl.load(x_ptr + batch_idx * input_size + offsets, mask=mask, other=0.0)
        # Load precomputed weight sum
        w_sum = tl.load(weight_sum_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate the dot product
        acc += tl.sum(x * w_sum, axis=0)
    
    # Apply scaling factor (includes division by 2)
    out_val = acc * scale
    
    # Store the result
    tl.store(out_ptr + batch_idx, out_val)


def fused_matmul_sum_scale(x: torch.Tensor, weight_sum: torch.Tensor, scale: float):
    """
    Fused kernel that performs:
    1. Matrix multiplication with weight
    2. Division by 2
    3. Sum across hidden dimension
    4. Scaling by scaling_factor
    
    All in a single kernel launch.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    weight_sum = weight_sum.contiguous()
    
    batch_size, input_size = x.shape
    
    # Prepare output tensor (batch_size, 1)
    out = torch.empty((batch_size, 1), dtype=x.dtype, device=x.device)
    
    # Block size tuning parameter
    BLOCK_SIZE = 256
    
    # Determine the number of blocks (one per batch element)
    grid = (batch_size,)
    
    # Launch the Triton kernel
    fused_matmul_sum_scale_kernel[grid](
        x,
        weight_sum,
        out,
        batch_size,
        input_size,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses matrix multiplication, division, summation, and scaling
    into a single Triton kernel. Uses precomputed weight sum to reduce computation.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        # Original weight parameter
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        
        # Precompute weight sum across hidden dimension for optimization
        # This allows us to avoid the full GEMM o
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 14
```
