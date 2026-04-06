# KernelBench Level 2 Problem 92: 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp.py

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
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization
        x_norm = self.group_norm(x_conv)
        # Tanh
        x_tanh = self.tanh(x_norm)
        # HardSwish
        x_hard_swish = self.hard_swish(x_tanh)
        # Residual Addition
        x_res = x_conv + x_hard_swish
        # LogSumExp
        x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
groups = 16

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]
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

A previous single-shot attempt failed. Error: `at 61:15:
    var = tl.sum(diff * diff, axis=0) / channels_per_group

    # Load gamma and beta
    gammas = tl.load(gamma_ptr + c, mask=c_mask, other=1.0)
    betas = tl.load(beta_ptr + c, mask=c_...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_activation_residual_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    num_groups,
    eps,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel: GroupNorm + Tanh + HardSwish + Residual Addition
    Each program handles one (batch, group, height, width) location
    """
    pid = tl.program_id(0)
    
    spatial_size = H * W
    per_batch = num_groups * spatial_size
    
    batch_idx = pid // per_batch
    remainder = pid % per_batch
    group_idx = remainder // spatial_size
    spatial_idx = remainder % spatial_size
    height_idx = spatial_idx // W
    width_idx = spatial_idx % W
    
    channels_per_group = C // num_groups
    group_start = group_idx * channels_per_group
    
    # Load all channels in this group
    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < channels_per_group
    c = group_start + c_offsets
    
    ptrs = x_ptr + batch_idx * stride_n + c * stride_c + height_idx * stride_h + width_idx * stride_w
    x_vals = tl.load(ptrs, mask=c_mask, other=0.0)
    
    # Compute mean across channels in group
    mean = tl.sum(x_vals, axis=0) / channels_per_group
    
    # Compute variance
    diff = x_vals - mean
    var = tl.sum(diff * diff, axis=0) / channels_per_group
    
    # Load gamma and beta
    gammas = tl.load(gamma_ptr + c, mask=c_mask, other=1.0)
    betas = tl.load(beta_ptr + c, mask=c_mask, other=0.0)
    
    # GroupNorm
    normalized = (x_vals - mean) / tl.sqrt(var + eps)
    normalized = normalized * gammas + betas
    
    # Tanh
    tanh_out = tl.libdevice.tanh(normalized)
    
    # HardSwish: x * relu6(x + 3) / 6
    hard_swish_out = tanh_out * tl.minimum(tl.maximum(tanh_out + 3.0, 0.0), 6.0) / 6.0
    
    # Residual addition
    residual = x_vals + hard_swish_out
    
    # Store output
    out_ptrs = out_ptr + batch_idx * stride_n + c * stride_c + height_idx * stride_h + width_idx * stride_w
    tl.store(out_ptrs, residual, mask=c_mask)


@triton.jit
def logsumexp_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    BLOCK_C: tl.constexpr,
):
    """
    LogSumExp across channels with numerical stability
    Each program handles one (batch, height, width) location
    """
    pid = tl.program_id(0)
    batch_idx = pid // (H * W)
    hw_idx = pid % (H * W)
    height_idx = hw_idx // W
    width_idx = hw_idx % W
    
    # Load all channels
    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    ptrs = x_ptr + batch_idx * stride_n + c_offsets * stride_c + height_idx * stride_h + width_idx * stride_w
    vals = tl.load(ptrs, mask=c_mask, other=-1e30)
    
    # LogSumExp: max + log(sum(exp(x - max)))
    max_val = tl.max(vals, axis=0)
    exp_vals = tl.exp(vals - max_val)
    sum
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 92
```
