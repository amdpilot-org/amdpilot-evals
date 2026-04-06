# KernelBench Level 1 Problem 11: 11_4D_tensor_matrix_multiplication.py

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
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return torch.einsum("bijl,lk->bijk", A, B)

# Test code
b = 8
i = 256
j = 512
l = 256
k = 768

def get_inputs():
    A = torch.rand(b, i, j, l)
    B = torch.rand(l, k)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
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

A previous single-shot attempt failed. Error: `at 58:19:
                     i_idx * stride_a_i + 
                     j_idx * stride_a_j + 
                     l_offs * stride_a_l)
            a = tl.load(A_ptr + a_offs, mask=l_mask, other=...`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tensor_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    batch_size,
    i_size,
    j_size,
    L: tl.constexpr,
    K: tl.constexpr,
    stride_a_batch,
    stride_a_i,
    stride_a_j,
    stride_a_l,
    stride_b_l,
    stride_b_k,
    stride_c_batch,
    stride_c_i,
    stride_c_j,
    stride_c_k,
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (b, i, j) position and computes all k values
    pid = tl.program_id(0)
    
    # Decode pid to (b, i, j)
    j_idx = pid % j_size
    ij_idx = pid // j_size
    i_idx = ij_idx % i_size
    b_idx = ij_idx // i_size
    
    # For each k position, compute the dot product over l
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        
        # Accumulator for this block of k values
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        
        # Iterate over l dimension in blocks
        for l_start in range(0, L, BLOCK_L):
            l_offs = l_start + tl.arange(0, BLOCK_L)
            l_mask = l_offs < L
            
            # Load from A: shape (BLOCK_L,)
            a_offs = (b_idx * stride_a_batch + 
                     i_idx * stride_a_i + 
                     j_idx * stride_a_j + 
                     l_offs * stride_a_l)
            a = tl.load(A_ptr + a_offs, mask=l_mask, other=0.0)
            
            # Load from B: shape (BLOCK_L, BLOCK_K)
            b_offs = (l_offs[:, None] * stride_b_l + 
                     k_offs[None, :] * stride_b_k)
            b = tl.load(B_ptr + b_offs, mask=l_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Matrix multiply: (BLOCK_L,) @ (BLOCK_L, BLOCK_K) -> (BLOCK_K,)
            acc += tl.dot(a[:, None], b)
        
        # Store result
        c_offs = (b_idx * stride_c_batch + 
                 i_idx * stride_c_i + 
                 j_idx * stride_c_j + 
                 k_offs * stride_c_k)
        tl.store(C_ptr + c_offs, acc, mask=k_mask)


def triton_tensor_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 4D tensor-matrix multiplication using Triton kernel:
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Args:
        A: Input 4D tensor of shape (b, i, j, l)
        B: Input matrix of shape (l, k)
    
    Returns:
        Output 4D tensor of shape (b, i, j, k)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Tensors must be FP32."
    
    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    b, i, j, l = A.shape
    k = B.shape[1]
    
    # Verify dimensions match
    assert l == B.shape[0], f"Dimension mismatch: A has l={l}, B has l={B.shape[0]}"
    
    # Prepare output tensor
    C = torch.empty((b, i, j, k), dtype=torch.float32, device=A.device)
    
    # Block s
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 11
```
