# KernelBench Level 2 Problem 51: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd.py

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
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm
        x = self.gemm(x)

        # Subtract
        x = x - self.subtract

        # GlobalAvgPool
        x = torch.mean(x, dim=1, keepdim=True)

        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)

        # GELU
        x = torch.nn.functional.gelu(x)

        # ResidualAdd
        x = x + original_x

        return x

batch_size = 2048
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

A previous single-shot attempt failed. Error: `error: HIP error: invalid argument
Search for `hipErrorInvalidValue' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__HIPRT__TYPES.html for more information.
HIP kernel errors might be asynchro`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_subtract_kernel(
    x_ptr,
    subtract_ptr,
    out_ptr,
    batch_size,
    n_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise subtract with broadcasting: out = x - subtract
    x: (batch_size, n_features)
    subtract: (n_features,)
    out: (batch_size, n_features)
    """
    pid = tl.program_id(0)
    batch_idx = pid // n_features
    feat_idx = pid % n_features
    
    if batch_idx < batch_size and feat_idx < n_features:
        x_val = tl.load(x_ptr + batch_idx * n_features + feat_idx)
        sub_val = tl.load(subtract_ptr + feat_idx)
        out_val = x_val - sub_val
        tl.store(out_ptr + batch_idx * n_features + feat_idx, out_val)


@triton.jit
def fused_reduce_gelu_residual_kernel(
    x_ptr,
    original_x_ptr,
    out_ptr,
    batch_size,
    n_features,
    stride_x,
    stride_orig,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: GlobalAvgPool + LogSumExp + GELU + ResidualAdd
    Input x: (batch_size, n_features)
    Input original_x: (batch_size, n_features)
    Output: (batch_size, n_features)
    """
    pid = tl.program_id(0)
    
    if pid < batch_size:
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_features
        
        # Load row for reduction
        x_ptrs = x_ptr + pid * stride_x + offs
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # GlobalAvgPool: compute mean across features
        mean = tl.sum(x, axis=0) / n_features
        
        # LogSumExp on single value is identity (log(exp(mean)) = mean)
        val = mean
        
        # GELU activation
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_cubed = val * val * val
        inner = 0.7978845608 * (val + 0.044715 * x_cubed)
        gelu_val = 0.5 * val * (1.0 + tl.libdevice.tanh(inner))
        
        # Load original_x and add residual (broadcasting gelu_val)
        orig_ptrs = original_x_ptr + pid * stride_orig + offs
        original = tl.load(orig_ptrs, mask=mask, other=0.0)
        
        out = original + gelu_val
        
        # Store output
        out_ptrs = out_ptr + pid * stride_out + offs
        tl.store(out_ptrs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized Model with Triton kernels
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        original_x = x.clone().detach()
        batch_size = x.shape[0]
        
        # Gemm (keep PyTorch's optimized cuBLAS implementation)
        x = self.gemm(x)
        
        # Fused Subtract kernel
        subtra
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 51
```
