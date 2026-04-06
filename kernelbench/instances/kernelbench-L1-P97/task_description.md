# KernelBench Level 1 Problem 97: 97_ScaledDotProductAttention.py

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
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        return out

batch_size = 32
num_heads = 32
sequence_length = 512
embedding_dimension = 1024

def get_inputs():
    Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension)
    return [Q, K, V]

def get_init_inputs():
    return []

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

A previous single-shot attempt failed. Error: `at 38:13:

    for start_n in range(0, seq_len, BLOCK_N):
        n_range = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_range < seq_len

        q = tl.load(Q_ptr + q_offset + tl.arange(0, B...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    num_heads, seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_queries = num_heads * seq_len
    query_idx = pid % seq_len
    head_idx = (pid // seq_len) % num_heads
    batch_idx = pid // (num_heads * seq_len)
    
    q_offset = batch_idx * stride_qb + head_idx * stride_qh + query_idx * stride_qs
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    out_offset = batch_idx * stride_ob + head_idx * stride_oh + query_idx * stride_os
    
    m_i = tl.zeros([BLOCK_D], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for start_n in range(0, seq_len, BLOCK_N):
        n_range = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_range < seq_len
        
        q = tl.load(Q_ptr + q_offset + tl.arange(0, BLOCK_D) * stride_qd, 
                    mask=tl.arange(0, BLOCK_D) < head_dim, other=0.0).to(tl.float32)
        
        k = tl.load(K_ptr + k_offset + n_range[:, None] * stride_ks + tl.arange(0, BLOCK_D)[None, :] * stride_kd,
                    mask=(n_mask[:, None]) & (tl.arange(0, BLOCK_D)[None, :] < head_dim), other=0.0).to(tl.float32)
        
        qk = tl.dot(q[None, :], k, allow_tf32=False) * scale
        qk = tl.where(n_mask[:, None], qk, float('-inf'))
        
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        
        v = tl.load(V_ptr + v_offset + n_range[:, None] * stride_vs + tl.arange(0, BLOCK_D)[None, :] * stride_vd,
                    mask=(n_mask[:, None]) & (tl.arange(0, BLOCK_D)[None, :] < head_dim), other=0.0).to(tl.float32)
        
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
        l_i = l_new
    
    out = acc / l_i[:, None]
    tl.store(Out_ptr + out_offset + tl.arange(0, BLOCK_D) * stride_od, out, 
             mask=tl.arange(0, BLOCK_D) < head_dim)


def triton_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Tensors must be on CUDA."
    assert Q.dtype == torch.float32, "Input must be FP32"
    
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    batch_size, num_heads, seq_len, head_dim = Q.shape
    out = torch.empty_like(Q)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 97
```
