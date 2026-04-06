# KernelBench Level 2 Problem 66: 66_Matmul_Dropout_Softmax.py

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
    A model that performs matrix multiplication, applies dropout, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = self.dropout(x)
        x = torch.softmax(x, dim=1)  # Softmax over features
        return x

batch_size = 128
in_features = 16384
out_features = 16384
dropout_p = 0.2

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]

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
def fused_dropout_softmax_kernel(
    x_ptr,
    out_ptr,
    stride_x_batch,
    stride_x_feat,
    stride_out_batch,
    stride_out_feat,
    batch_size,
    n_features,
    dropout_p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_x = pid * stride_x_batch
    row_start_out = pid * stride_out_batch

    # First pass: compute max and sum of exp for numerical stability
    max_val = -float('inf')
    sum_exp = 0.0

    for block_start in range(0, n_features, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        x = tl.load(x_ptr + row_start_x + offsets * stride_x_feat, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        max_val = new_max

    # Second pass: apply softmax and dropout
    for block_start in range(0, n_features, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_features
        x = tl.load(x_ptr + row_start_x + offsets * stride_x_feat, mask=mask, other=0.0)
        exp_x = tl.exp(x - max_val)
        softmax_out = exp_x / sum_exp
        rand_vals = tl.rand(seed, pid * n_features + offsets)
        dropout_mask = rand_vals > dropout_p
        out = softmax_out * dropout_mask / (1.0 - dropout_p)
        tl.store(out_ptr + row_start_out + offsets * stride_out_feat, out, mask=mask)


def fused_dropout_softmax(x: torch.Tensor, dropout_p: float, seed: int = None):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size, n_features = x.shape
    out = torch.empty_like(x)

    if seed is None:
        seed = torch.randint(0, 2**31, (1,)).item()

    BLOCK_SIZE = 1024
    grid = (batch_size,)

    fused_dropout_softmax_kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        batch_size,
        n_features,
        dropout_p,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x):
        x = self.matmul(x)
        x = fused_dropout_softmax(x, self.dropout_p)
        return x
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 66
```
