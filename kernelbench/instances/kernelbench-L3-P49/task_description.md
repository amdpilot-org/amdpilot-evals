# KernelBench Level 3 Problem 49: 49_Mamba2ReturnFinalState.py

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
from einops import rearrange

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def segsum(self, x):
        """Naive segment sum calculation."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs
        L = torch.exp(self.segsum(A_blocks))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        return new_states[:, -1]

# Test parameters
batch_size = 2048
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.rand(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]

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

A previous single-shot attempt failed. Error: `dynamic_func() got multiple values for argument 'BLOCK_SIZE'`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl


@triton.jit
def cumsum_exp_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_c,
    stride_h,
    stride_l,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < L
    
    x_ptrs = x_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h + offsets * stride_l
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    cumsum = tl.cumsum(x, axis=0)
    exp_cumsum = tl.exp(cumsum)
    
    out_ptrs = out_ptr + pid_b * stride_b + pid_c * stride_c + pid_h * stride_h + offsets * stride_l
    tl.store(out_ptrs, exp_cumsum, mask=mask)


@triton.jit
def segsum_exp_kernel(
    x_ptr,
    out_ptr,
    stride_b,
    stride_h,
    stride_c,
    stride_l,
    stride_l2,
    B: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    row_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < L
    col_mask = col_offsets < L
    
    x_base = x_ptr + pid_b * stride_b + pid_h * stride_h + pid_c * stride_c
    x = tl.load(x_base + row_offsets * stride_l, mask=row_mask, other=0.0)
    cumsum = tl.cumsum(x, axis=0)
    
    cumsum_row = tl.expand_dims(cumsum, 1)
    cumsum_col = tl.expand_dims(cumsum, 0)
    segsum = cumsum_row - cumsum_col
    
    mask_matrix = row_offsets[:, None] >= col_offsets[None, :]
    segsum = tl.where(mask_matrix, segsum, -1e10)
    
    exp_segsum = tl.exp(segsum)
    
    out_base = out_ptr + pid_b * stride_b + pid_h * stride_h + pid_c * stride_c
    out_ptrs = out_base + row_offsets[:, None] * stride_l + col_offsets[None, :] * stride_l2
    tl.store(out_ptrs, exp_segsum, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def diag_block_kernel(
    C_ptr,
    B_ptr,
    L_ptr,
    X_ptr,
    Y_ptr,
    stride_b,
    stride_c,
    stride_l,
    stride_h,
    stride_n,
    stride_p,
    B_SIZE: tl.constexpr,
    C_SIZE: tl.constexpr,
    L_SIZE: tl.constexpr,
    H_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    P_SIZE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_p = tl.program_id(4)
    
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = l_offsets < L_SIZE
    
    s_offsets = tl.arange(0, BLOCK_L)
    s_mask = s_offsets < L_SIZE
    
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N_SIZE
    
    C_base = C_ptr + pid_b * stride_b + pid_c * stride_c + l_offsets[:, None] * stride_l + pid_h * st
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 49
```
