# KernelBench Level 2 Problem 19: 19_ConvTranspose2d_GELU_GroupNorm.py

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
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.nn.functional.gelu(x)
        x = self.group_norm(x)
        return x

batch_size   = 128  
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride       = 1
groups = 8
num_groups = 8

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]
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

A previous single-shot attempt failed. Error: `at 73:48:
        b_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

        for h in range(H):
            for w in range(W):
                idx = n * C * H * W + c_idx * H *...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_groupnorm_fused_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for GELU activation + GroupNorm.
    Each program handles one sample and one group of channels.
    """
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Compute mean and variance for this group
    mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for c_off in range(0, channels_per_group, BLOCK_SIZE):
        c_idx = group_start + c_off + tl.arange(0, BLOCK_SIZE)
        mask = c_idx < group_start + channels_per_group
        
        # Accumulate sum and sum of squares across H*W
        sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for h in range(H):
            for w in range(W):
                idx = n * C * H * W + c_idx * H * W + h * W + w
                x_val = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
                sum_val += x_val
                sum_sq += x_val * x_val
        
        hw = H * W
        mean += sum_val / hw
        var += sum_sq / hw

    # Reduce mean and var across the block
    mean = tl.sum(mean) / channels_per_group
    var = tl.sum(var) / channels_per_group
    var = var - mean * mean
    var = var + eps
    rstd = 1.0 / tl.sqrt(var)

    # Apply GELU + GroupNorm
    for c_off in range(0, channels_per_group, BLOCK_SIZE):
        c_idx = group_start + c_off + tl.arange(0, BLOCK_SIZE)
        mask = c_idx < group_start + channels_per_group
        
        # Load weight and bias for this channel
        w_val = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
        b_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

        for h in range(H):
            for w in range(W):
                idx = n * C * H * W + c_idx * H * W + h * W + w
                
                x_val = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
                
                # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x_cubed = x_val * x_val * x_val
                tanh_arg = 0.7978845608 * (x_val + 0.044715 * x_cubed)
                gelu_out = 0.5 * x_val * (1.0 + tl.libdevice.tanh(tanh_arg))
                
                # GroupNorm
                normalized = (gelu_out - mean) * rstd
                out_val = normalized * w_val + b_val
                
                tl.store(out_ptr + idx, out_val, mask=mask)


def gelu_groupnorm_fused(x, weight, bias, num_groups, eps=1e-5):
    """
    Fused GELU + GroupNorm operation using Triton.
    """
    a
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 19
```
