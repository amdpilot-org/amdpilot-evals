# KernelBench Level 3 Problem 2: 2_ShallowWideMLP.py

## Goal

Write an optimized Triton kernel implementation (`ModelNew`) that:
1. Produces the **exact same output** as the PyTorch reference `Model`
2. Is **faster** than the PyTorch baseline
3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)

## PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)

# Test code
batch_size = 128
input_size = 16384
hidden_layer_sizes = [32768, 32768]
output_size = 16384

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
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
def fused_linear_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,  # batch_size
    N,  # input_features
    K,  # output_features
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_om,
    stride_ok,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    m_range = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_range = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_range < M
    n_mask = n_range < K
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_range < N
        
        x_ptrs = x_ptr + m_range[:, None] * stride_xm + k_range[None, :] * stride_xn
        x = tl.load(x_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)
        
        w_ptrs = weight_ptr + n_range[:, None] * stride_wk + k_range[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(n_mask[:, None] & k_mask[None, :]), other=0.0)
        
        acc += tl.dot(x, tl.trans(w))
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0)
        acc += bias[None, :]
    
    acc = tl.maximum(acc, 0.0)
    
    out_ptrs = out_ptr + m_range[:, None] * stride_om + n_range[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,  # batch_size
    N,  # input_features
    K,  # output_features
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_om,
    stride_ok,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    m_range = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_range = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_range < M
    n_mask = n_range < K
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_range < N
        
        x_ptrs = x_ptr + m_range[:, None] * stride_xm + k_range[None, :] * stride_xn
        x = tl.load(x_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)
        
        w_ptrs = weight_ptr + n_range[:, None] * stride_wk + k_range[None, :] * stride_wn
        w = tl.load(w_ptrs, mask=(n_mask[:, None] & k_mask[None, :]), other=0.0)
        
        acc += tl.dot(x, tl.trans(w))
    
    if HAS_BIAS:
       
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 2
```
