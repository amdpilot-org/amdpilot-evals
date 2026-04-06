# KernelBench Level 2 Problem 94: 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm.py

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
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = x + self.bias
        x = self.hardtanh(x)
        x = self.mish(x)
        x = self.groupnorm(x)
        return x


batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)
num_groups = 256

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape, num_groups]
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

A previous single-shot attempt failed. Error: `at 32:15:

    # Add bias
    x = x + bias

    # Hardtanh: clamp between -1 and 1
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)

    # Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_activation_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_x,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses bias addition, hardtanh, and mish activations.
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input and bias
    x = tl.load(x_ptr + row_idx * stride_x + col_offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)

    # Add bias
    x = x + bias

    # Hardtanh: clamp between -1 and 1
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)

    # Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    # For numerical stability
    softplus = tl.where(x > 20, x, tl.log(1.0 + tl.exp(x)))
    mish = x * tl.libdevice.tanh(softplus)

    # Store output
    tl.store(out_ptr + row_idx * stride_out + col_offsets, mish, mask=mask)


@triton.jit
def group_norm_fwd_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    n_rows,
    n_cols,
    num_groups,
    stride_x,
    stride_out,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Implements group normalization for 2D input (N, C).
    Each row is normalized independently across channels within each group.
    """
    row_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    channels_per_group = n_cols // num_groups
    col_start = group_idx * channels_per_group

    # Compute mean and variance for this group
    acc_mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(0, channels_per_group, BLOCK_SIZE):
        col_offsets = col_start + i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < (col_start + channels_per_group)
        x = tl.load(x_ptr + row_idx * stride_x + col_offsets, mask=mask, other=0.0)
        acc_mean += x
        acc_var += x * x

    mean = tl.sum(acc_mean, axis=0) / channels_per_group
    var = tl.sum(acc_var, axis=0) / channels_per_group - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine transform
    for i in range(0, channels_per_group, BLOCK_SIZE):
        col_offsets = col_start + i + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < (col_start + channels_per_group)
        x = tl.load(x_ptr + row_idx * stride_x + col_offsets, mask=mask, other=0.0)
        x_norm = (x - mean) * rstd

        if weight_ptr is not None:
            weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
            x_norm = x_norm * weight

        if bias_ptr is not None:
            bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
            x_norm = x_norm + bias

        tl.store(out_ptr + row_idx * stride_out + col_offsets, x_norm, mask=mask)


def fused_bias_activation(x: torch.Tensor, bias: torch.Tensor):
    """
    Wra
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 94
```
