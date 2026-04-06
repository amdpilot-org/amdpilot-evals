# KernelBench Level 1 Problem 35: 35_GroupNorm_.py

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
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)

batch_size = 112  # scaled up
features = 64
num_groups = 8
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups] # num_features
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

A previous single-shot attempt failed. Error: `at 100:4:
        )

        x_vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
        normalized = (x_vals - mean) * rstd

        gamma_vals = tl.load(gamma_ptr + c_offsets, mask=mask, other=1...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_fwd_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    num_groups,
    num_channels,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    channels_per_group = num_channels // num_groups
    c_start = g * channels_per_group

    group_size = channels_per_group * H * W

    mean_acc = tl.zeros([1], dtype=tl.float32)
    var_acc = tl.zeros([1], dtype=tl.float32)

    for i in range(0, group_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size

        c_offsets = c_start + (offsets // (H * W))
        hw_offsets = offsets % (H * W)
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets % W

        ptrs = (
            n * stride_n +
            c_offsets * stride_c +
            h_offsets * stride_h +
            w_offsets * stride_w
        )

        x_vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
        mean_acc += tl.sum(x_vals, axis=0)

    mean = mean_acc / group_size

    for i in range(0, group_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size

        c_offsets = c_start + (offsets // (H * W))
        hw_offsets = offsets % (H * W)
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets % W

        ptrs = (
            n * stride_n +
            c_offsets * stride_c +
            h_offsets * stride_h +
            w_offsets * stride_w
        )

        x_vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
        diff = x_vals - mean
        var_acc += tl.sum(diff * diff, axis=0)

    variance = var_acc / group_size
    rstd = 1.0 / tl.sqrt(variance + eps)

    for i in range(0, group_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < group_size

        c_offsets = c_start + (offsets // (H * W))
        hw_offsets = offsets % (H * W)
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets % W

        ptrs = (
            n * stride_n +
            c_offsets * stride_c +
            h_offsets * stride_h +
            w_offsets * stride_w
        )

        x_vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
        normalized = (x_vals - mean) * rstd

        gamma_vals = tl.load(gamma_ptr + c_offsets, mask=mask, other=1.0)
        beta_vals = tl.load(beta_ptr + c_offsets, mask=mask, other=0.0)

        out_vals = normalized * gamma_vals + beta_vals
        tl.store(out_ptr + ptrs, out_vals, mask=mask)

    tl.store(mean_ptr + n * num_groups + g, mean)
    tl.store(rstd_ptr + n * num_groups + g, rstd)


def triton_group_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5
) -> 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 35
```
