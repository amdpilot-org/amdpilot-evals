# KernelBench Level 2 Problem 33: 33_Gemm_Scale_BatchNorm.py

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
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
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

A previous single-shot attempt failed. Error: `at 29:17:
    IS_TRAINING: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_n = pid_n * BLOCK_SIZE_N
    offs_n = block_st...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gemm_scale_kernel(
    a_ptr,
    w_ptr,
    bias_ptr,
    scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_a_m,
    stride_a_k,
    stride_w_n,
    stride_w_k,
    stride_out_m,
    stride_out_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k_curr = k + offs_k
        mask_k = offs_k_curr < K
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_a_m + offs_k_curr[None, :] * stride_a_k
        w_ptrs = w_ptr + offs_n[None, :] * stride_w_n + offs_k_curr[:, None] * stride_w_k
        
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(a, w)
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]
    
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0)
    acc *= scale[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def batchnorm_forward_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    M,
    N,
    stride_x_m,
    stride_x_n,
    stride_out_m,
    stride_out_n,
    eps,
    momentum,
    IS_TRAINING: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    block_start_n = pid_n * BLOCK_SIZE_N
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    
    if IS_TRAINING:
        offs_m = tl.arange(0, M)
        mask_m = offs_m < M
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_x_m + offs_n[None, :] * stride_x_n
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        
        mean = tl.sum(x, axis=0) / M
        diff = x - mean[None, :]
        var = tl.sum(diff * diff, axis=0) / M
        
        if pid_m == 0:
            tl.store(mean_ptr + offs_n, mean, mask=mask_n)
            tl.store(var_ptr + offs_n, var, mask=mask_n)
            
            if running_mean_ptr is not None:
                running_mean = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0)
                new_running
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 33
```
