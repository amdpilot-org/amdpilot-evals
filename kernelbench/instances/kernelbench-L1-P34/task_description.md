# KernelBench Level 1 Problem 34: 34_InstanceNorm.py

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
    Simple model that performs Instance Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)

batch_size = 112  # heavier workload
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

A previous single-shot attempt failed. Error: `at 62:26:
    # Pass 2: Compute variance
    var_acc = 0.0
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_eleme...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def instance_norm_fwd_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    N,
    C,
    H,
    W,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    # Base offset for this (n, c)
    base_offset = n * stride_n + c * stride_c
    
    # Number of spatial elements
    num_elements = H * W
    
    # Pass 1: Compute mean
    mean_acc = 0.0
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        
        # Convert 1D offset to 2D (h, w)
        h = offsets // W
        w = offsets % W
        
        x_offsets = base_offset + h * stride_h + w * stride_w
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        mean_acc += tl.sum(x_vals, axis=0)
    
    mean = mean_acc / num_elements
    
    # Store mean
    tl.store(mean_ptr + n * C + c, mean)
    
    # Pass 2: Compute variance
    var_acc = 0.0
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        
        h = offsets // W
        w = offsets % W
        
        x_offsets = base_offset + h * stride_h + w * stride_w
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        var_acc += tl.sum((x_vals - mean) ** 2, axis=0)
    
    var = var_acc / num_elements
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store rstd
    tl.store(rstd_ptr + n * C + c, rstd)
    
    # Pass 3: Normalize and apply affine
    for start in range(0, num_elements, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        
        h = offsets // W
        w = offsets % W
        
        x_offsets = base_offset + h * stride_h + w * stride_w
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        normalized = (x_vals - mean) * rstd
        
        if HAS_WEIGHT:
            w_val = tl.load(weight_ptr + c)
            normalized = normalized * w_val
        
        if HAS_BIAS:
            b_val = tl.load(bias_ptr + c)
            normalized = normalized + b_val
        
        tl.store(out_ptr + x_offsets, normalized, mask=mask)


def triton_instance_norm(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Triton-based Instance Normalization.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        weight: Optional weight tensor of shape (C,)
        bias: Optional bias tensor of shape (C,)
        eps: Epsilon for numerical stability
    
 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 34
```
