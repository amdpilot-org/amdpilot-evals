# KernelBench Level 2 Problem 97: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish.py

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
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)
        x = x + self.bias
        x = x / self.divide_value
        x = x * torch.sigmoid(x)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
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
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_bias_div_swish_kernel(
    x_ptr,
    out_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    extra_bias_ptr,
    divide_value,
    n_rows,
    n_cols,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    x = tl.load(x_ptr + row_idx * stride_x + col_offsets, mask=mask, other=0.0)
    mean = tl.load(running_mean_ptr + col_offsets, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + col_offsets, mask=mask, other=0.0)
    bn_weight = tl.load(bn_weight_ptr + col_offsets, mask=mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + col_offsets, mask=mask, other=0.0)
    extra_bias = tl.load(extra_bias_ptr)
    
    eps = 1e-5
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_bn = x_norm * bn_weight + bn_bias
    
    x = x_bn + extra_bias
    x = x / divide_value
    
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    x = x * sigmoid_x
    
    tl.store(out_ptr + row_idx * stride_out + col_offsets, x, mask=mask)


def fused_bn_bias_div_swish(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    extra_bias: torch.Tensor,
    divide_value: float,
):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, n_cols = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    grid = (batch_size,)
    
    fused_bn_bias_div_swish_kernel[grid](
        x,
        out,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        extra_bias,
        divide_value,
        batch_size,
        n_cols,
        x.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication followed by 
    fused batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, x):
        x = self.matmul(x)
        
        self.bn.eval()
        
        x = fused_bn_bias_div_swish(
            x,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.bias,
            self.divide_value,
        )
        
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 97
```
