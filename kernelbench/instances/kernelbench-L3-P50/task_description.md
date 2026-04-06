# KernelBench Level 3 Problem 50: 50_ReLUSelfAttention.py

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
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Model(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.relu(att)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y

batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [torch.rand(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
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

A previous single-shot attempt failed. Error: `at 34:12:
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, T, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < T

        k_...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vk,
    stride_ob, stride_oh, stride_ot, stride_ok,
    B, H, T, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < T
    
    q_ptrs = (pid_b * stride_qb + pid_h * stride_qh + 
              m_offs[:, None] * stride_qt + tl.arange(0, BLOCK_D)[None, :] * stride_qk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    for n_start in range(0, T, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < T
        
        k_ptrs = (pid_b * stride_kb + pid_h * stride_kh + 
                  n_offs[:, None] * stride_kt + tl.arange(0, BLOCK_D)[None, :] * stride_kk)
        v_ptrs = (pid_b * stride_vb + pid_h * stride_vh + 
                  n_offs[:, None] * stride_vt + tl.arange(0, BLOCK_D)[None, :] * stride_vk)
        
        q = tl.load(q_ptrs, mask=m_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D), other=0.0)
        k = tl.load(k_ptrs, mask=n_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D), other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D), other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) / tl.sqrt(D)
        
        causal_mask = m_offs[:, None] >= n_offs[None, :]
        qk = tl.where(causal_mask, qk, float('-inf'))
        
        qk = tl.maximum(qk, 0.0)
        
        acc += tl.dot(qk, v)
    
    out_ptrs = (pid_b * stride_ob + pid_h * stride_oh + 
                m_offs[:, None] * stride_ot + tl.arange(0, BLOCK_D)[None, :] * stride_ok)
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < D))


def triton_fused_attention(q, k, v):
    """
    Fused attention kernel: (Q @ K^T) * scale + mask + ReLU @ V
    """
    B, H, T, D = q.shape
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = 64
    
    out = torch.empty_like(q)
    
    grid = (B, H, (T + BLOCK_M - 1) // BLOCK_M)
    
    fused_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for fused attention computation.
    """

    def __init__(s
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 50
```
