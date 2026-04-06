# KernelBench Level 3 Problem 31: 31_VisionAttention.py

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
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)  # (seq_len, batch_size, embed_dim)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    return [embed_dim, num_heads]
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

A previous single-shot attempt failed. Error: `at 19:15:
    row_start = pid * embed_dim

    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < n_elements

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    batch_size, seq_len, num_heads, head_dim,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < seq_len
    
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - 1e6
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    q_ptrs = Q_ptr + pid_b * stride_qb + m_offsets[:, None] * stride_qs + pid_h * stride_qh + tl.arange(0, BLOCK_D)[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    
    for n_start in range(0, seq_len, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < seq_len
        
        k_ptrs = K_ptr + pid_b * stride_kb + n_offsets[None, :] * stride_ks + pid_h * stride_kh + tl.arange(0, BLOCK_D)[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)
        
        v_ptrs = V_ptr + pid_b * stride_vb + n_offsets[:, None] * stride_vs + pid_h * stride_vh + tl.arange(0, BLOCK_D)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        
        qk = tl.dot(q, k) * scale
        
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        m_i = m_new
        l_i = l_new
    
    acc = acc / l_i[:, None]
    
    out_ptrs = Out_ptr + pid_b * stride_ob + m_offsets[:, None] * stride_os + pid_h * stride_oh + tl.arange(0, BLOCK_D)[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def fused_residual_layernorm_kernel(
    x_ptr, attn_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements, embed_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * embed_dim
    
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < n_elements
    
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    attn = tl.load(attn_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    
    residual = x + attn
    
    row_base = pid * embed_dim
    row_mask = tl.arange(0, embed_dim) < embed_dim
    row_vals = tl.load(x_ptr + row_base + tl.arange(0, embed_dim), mask=row_mask, other=0.0).to(tl.float32)
    attn_vals = tl.load
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 31
```
