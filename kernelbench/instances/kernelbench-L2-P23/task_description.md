# KernelBench Level 2 Problem 23: 23_Conv3d_GroupNorm_Mean.py

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
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.conv(x)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4]) # Compute mean across all dimensions except batch
        return x

batch_size = 128
in_channels = 3
out_channels = 24
D, H, W = 24, 32, 32
kernel_size = 3
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups]
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
def groupnorm_mean_fused_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    B,
    C,
    D,
    H,
    W,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GroupNorm + Mean reduction kernel.
    Each program handles one sample in the batch.
    """
    batch_idx = tl.program_id(0)
    
    # Compute group size
    channels_per_group = C // num_groups
    
    # Pointer to the start of this batch
    batch_start = batch_idx * stride_b
    
    # Initialize accumulators for final mean
    total_sum = 0.0
    total_count = C * D * H * W
    
    # Process each group
    for group_id in range(num_groups):
        group_start_c = group_id * channels_per_group
        
        # First pass: compute mean and variance for this group
        group_sum = 0.0
        group_sum_sq = 0.0
        group_count = channels_per_group * D * H * W
        
        for c_offset in range(channels_per_group):
            c_idx = group_start_c + c_offset
            for d_idx in range(D):
                for h_idx in range(H):
                    for w_idx in range(W):
                        offset = (c_idx * stride_c + d_idx * stride_d + 
                                  h_idx * stride_h + w_idx * stride_w)
                        val = tl.load(x_ptr + batch_start + offset).to(tl.float32)
                        group_sum += val
                        group_sum_sq += val * val
        
        group_mean = group_sum / group_count
        group_var = group_sum_sq / group_count - group_mean * group_mean
        group_var = group_var + eps
        group_rstd = 1.0 / tl.sqrt(group_var)
        
        # Second pass: normalize and accumulate for final mean
        for c_offset in range(channels_per_group):
            c_idx = group_start_c + c_offset
            for d_idx in range(D):
                for h_idx in range(H):
                    for w_idx in range(W):
                        offset = (c_idx * stride_c + d_idx * stride_d + 
                                  h_idx * stride_h + w_idx * stride_w)
                        val = tl.load(x_ptr + batch_start + offset).to(tl.float32)
                        normalized = (val - group_mean) * group_rstd
                        tl.store(out_ptr + batch_start + offset, normalized)
                        total_sum += normalized
    
    # Compute final mean for this batch
    final_mean = total_sum / total_count
    tl.store(mean_ptr + batch_idx, final_mean)


@triton.jit
def groupnorm_mean_fused_kernel_optimized(
    x_ptr,
    out_ptr,
    mean_ptr,
    stride_b,
    stride_c,
    stride_d,
    stride_h,
    stride_w,
    B,
    C,
    D,
    H,
    W,
    num_groups,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Optim
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 23
```
