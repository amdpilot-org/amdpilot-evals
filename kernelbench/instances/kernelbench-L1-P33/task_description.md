# KernelBench Level 1 Problem 33: 33_BatchNorm.py

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
    Simple model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)

batch_size = 64
features = 64
dim1 = 512
dim2 = 512

def get_inputs():
    x = torch.rand(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]
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

A previous single-shot attempt failed. Error: `at 30:4:
    count = 0

    for block_start in range(0, num_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        actual_offs...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def batch_norm_compute_stats_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute mean and variance for each channel."""
    channel_id = tl.program_id(0)
    
    num_elements = N * H * W
    channel_offset = channel_id * H * W
    
    # Compute mean
    mean_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = 0
    
    for block_start in range(0, num_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        actual_offsets = channel_offset + offsets
        x_val = tl.load(x_ptr + actual_offsets, mask=mask, other=0.0)
        mean_sum += tl.sum(x_val, axis=0)
        count += tl.sum(mask.to(tl.int32))
    
    mean = mean_sum / count
    tl.store(mean_ptr + channel_id, mean)
    
    # Compute variance
    var_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, num_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        actual_offsets = channel_offset + offsets
        x_val = tl.load(x_ptr + actual_offsets, mask=mask, other=0.0)
        var_sum += tl.sum((x_val - mean) ** 2, axis=0)
    
    var = var_sum / count
    tl.store(var_ptr + channel_id, var)


@triton.jit
def batch_norm_apply_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply batch normalization using pre-computed statistics."""
    channel_id = tl.program_id(0)
    
    num_elements = N * H * W
    channel_offset = channel_id * H * W
    
    # Load statistics for this channel
    mean = tl.load(mean_ptr + channel_id)
    var = tl.load(var_ptr + channel_id)
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    
    # Compute normalization factor
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization and affine transform
    for block_start in range(0, num_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        actual_offsets = channel_offset + offsets
        
        x_val = tl.load(x_ptr + actual_offsets, mask=mask, other=0.0)
        normalized = (x_val - mean) * inv_std
        out_val = weight * normalized + bias
        
        tl.store(out_ptr + actual_offsets, out_val, mask=mask)


@triton.jit
def batch_norm_inference_kernel(
    x_ptr,
    out_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply batch normalization in inference mode using running statistics."""
    channel_id = tl.program_id(0)
    
    num_elements = N * H * W
    channel_offset = channel_id * H
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 33
```
