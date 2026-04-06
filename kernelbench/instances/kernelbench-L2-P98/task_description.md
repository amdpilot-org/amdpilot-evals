# KernelBench Level 2 Problem 98: 98_Matmul_AvgPool_GELU_Scale_Max.py

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
    A model implementing the pattern "Matmul_AvgPool_GELU_Scale_Max".
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.avg_pool(x.unsqueeze(1)).squeeze(1)
        x = torch.nn.functional.gelu(x)
        x = x * self.scale_factor
        x = torch.max(x, dim=1).values
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
pool_kernel_size = 16
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
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

A previous single-shot attempt failed. Error: `at 55:37:
            # Load with masking
            vals = tl.load(input_ptrs, mask=mask, other=0.0)
            pool_sums += vals

        # Compute average
        pool_avgs = pool_sums / pool_...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_avgpool_gelu_scale_max_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    in_features,
    pool_kernel_size,
    scale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Average pooling over groups of pool_kernel_size elements
    2. GELU activation
    3. Scale multiplication
    4. Max reduction over all pooled values
    
    Each program handles one batch element.
    """
    batch_idx = tl.program_id(0)
    
    # Base offset for this batch
    batch_offset = batch_idx * in_features
    
    # Number of pooled outputs
    num_pooled = in_features // pool_kernel_size
    
    # Initialize max value to negative infinity
    max_val = -1e30
    
    # Process pooled values in blocks
    for block_start in range(0, num_pooled, BLOCK_SIZE):
        # Create offsets for this block
        pooled_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = pooled_offsets < num_pooled
        
        # For each pooled position, compute average over pool_kernel_size elements
        # We'll load all elements for this block at once
        pool_sums = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for k in range(pool_kernel_size):
            # Calculate actual input offsets
            input_offsets = (pooled_offsets * pool_kernel_size + k)
            input_ptrs = x_ptr + batch_offset + input_offsets
            # Load with masking
            vals = tl.load(input_ptrs, mask=mask, other=0.0)
            pool_sums += vals
        
        # Compute average
        pool_avgs = pool_sums / pool_kernel_size
        
        # Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = pool_avgs
        x_cubed = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x_cubed)
        gelu_vals = 0.5 * x * (1.0 + tl.math.tanh(inner))
        
        # Apply scale factor
        scaled_vals = gelu_vals * scale_factor
        
        # Find max in this block
        block_max = tl.max(scaled_vals, axis=0, where=mask, return_indices=False)
        
        # Update global max
        if block_max > max_val:
            max_val = block_max
    
    # Store result for this batch
    tl.store(out_ptr + batch_idx, max_val)


def fused_avgpool_gelu_scale_max(x: torch.Tensor, pool_kernel_size: int, scale_factor: float):
    """
    Wrapper function for the fused kernel.
    
    Args:
        x: Input tensor of shape (batch_size, in_features)
        pool_kernel_size: Size of the average pooling window
        scale_factor: Multiplicative scale factor
    
    Returns:
        Output tensor of shape (batch_size,)
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()
    
    batch_size, in_features = x.shape
    
    # Prepare output tensor
    out = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    
    # Block
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 98
```
