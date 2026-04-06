# KernelBench Level 2 Problem 64: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py

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
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # LeakyReLU
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        # GELU
        x = torch.nn.functional.gelu(x)
        # GELU
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
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

A previous single-shot attempt failed. Error: `at 102:33:

    # Compute LogSumExp
    logsumexp = max_val + tl.log(logsumexp_sum)

    # Apply LeakyReLU twice (negative_slope=0.01)
    negative_slope = 0.01
    x_out = tl.where(logsumexp > 0, ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_linear_logsumexp_activations_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_features,
    out_features,
    has_bias: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
):
    """
    Fused kernel for Linear + LogSumExp + LeakyReLU (x2) + GELU (x2)
    Each program handles one batch element
    """
    batch_idx = tl.program_id(0)
    
    # Load input row
    in_row_start = batch_idx * in_features
    in_cols = tl.arange(0, BLOCK_SIZE_IN)
    in_mask = in_cols < in_features
    x_row = tl.load(x_ptr + in_row_start + in_cols, mask=in_mask, other=0.0)
    
    # First pass: compute all linear outputs and find max for LogSumExp stability
    max_val = -1e10
    linear_outputs = tl.zeros([BLOCK_SIZE_IN], dtype=tl.float32)
    
    # We'll compute linear outputs in chunks
    # For LogSumExp, we need all out_features values
    # We'll accumulate exp values directly
    
    logsumexp_sum = 0.0
    max_val = -1e10
    
    # Compute linear output for each feature and track max
    for out_feat in range(out_features):
        acc = 0.0
        weight_row_start = out_feat * in_features
        
        for start in range(0, in_features, BLOCK_SIZE_IN):
            weight_cols = tl.arange(0, BLOCK_SIZE_IN)
            weight_mask = (start + weight_cols) < in_features
            weight_segment = tl.load(
                weight_ptr + weight_row_start + start + weight_cols,
                mask=weight_mask,
                other=0.0
            )
            x_segment = tl.load(
                x_ptr + in_row_start + start + weight_cols,
                mask=weight_mask,
                other=0.0
            )
            acc += tl.sum(x_segment * weight_segment)
        
        if has_bias:
            bias_val = tl.load(bias_ptr + out_feat)
            acc += bias_val
        
        # Track max for numerical stability
        if acc > max_val:
            max_val = acc
        
        # Store for second pass (we'll recompute)
    
    # Second pass: compute LogSumExp with the max
    logsumexp_sum = 0.0
    for out_feat in range(out_features):
        acc = 0.0
        weight_row_start = out_feat * in_features
        
        for start in range(0, in_features, BLOCK_SIZE_IN):
            weight_cols = tl.arange(0, BLOCK_SIZE_IN)
            weight_mask = (start + weight_cols) < in_features
            weight_segment = tl.load(
                weight_ptr + weight_row_start + start + weight_cols,
                mask=weight_mask,
                other=0.0
            )
            x_segment = tl.load(
                x_ptr + in_row_start + start + weight_cols,
                mask=weight_mask,
                other=0.0
            )
            acc += tl.sum(x_segment * weight_segment)
        
        if has_bias:
            bias_val = tl.load(bias_ptr + out_feat)
            acc += 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 64
```
