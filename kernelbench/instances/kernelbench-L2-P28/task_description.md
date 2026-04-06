# KernelBench Level 2 Problem 28: 28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py

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
    Model that performs a batch matrix multiplication, instance normalization, summation, residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.instance_norm = nn.InstanceNorm2d(out_features, eps=eps, momentum=momentum)

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.bmm(x)
        x = self.instance_norm(x.unsqueeze(1).unsqueeze(1)).squeeze(1).squeeze(1)
        x = x + y
        x = x * y
        return x

batch_size = 1024  # Increased batch size
in_features = 8192  # Increased input features
out_features = 8192  # Increased output features

def get_inputs():
    return [torch.rand(batch_size, in_features), torch.rand(batch_size, out_features)]

def get_init_inputs():
    return [in_features, out_features]
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

A previous single-shot attempt failed. Error: `'NoneType' object has no attribute 'contiguous'`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def compute_mean_var_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    num_features: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute mean and variance for each sample across features.
    """
    sample_id = tl.program_id(0)
    
    base_offset = sample_id * num_features
    
    # Load all features for this sample
    feature_offsets = tl.arange(0, BLOCK_SIZE)
    mask = feature_offsets < num_features
    
    x = tl.load(x_ptr + base_offset + feature_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / num_features
    
    # Compute variance
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / num_features
    
    # Store mean and variance
    tl.store(mean_ptr + sample_id, mean)
    tl.store(var_ptr + sample_id, variance)


@triton.jit
def norm_add_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    num_features: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply normalization, add y, and multiply by y.
    """
    sample_id = tl.program_id(0)
    
    base_offset = sample_id * num_features
    
    # Load mean and variance for this sample
    mean = tl.load(mean_ptr + sample_id).to(tl.float32)
    variance = tl.load(var_ptr + sample_id).to(tl.float32)
    
    # Process features in blocks
    for block_start in range(0, num_features, BLOCK_SIZE):
        feature_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = feature_offsets < num_features
        
        # Load x and y
        x = tl.load(x_ptr + base_offset + feature_offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + base_offset + feature_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        x_norm = (x - mean) / tl.sqrt(variance + eps)
        
        # Apply weight and bias
        weight = tl.load(weight_ptr + feature_offsets, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(bias_ptr + feature_offsets, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm * weight + bias
        
        # Add y and multiply
        out = (x_norm + y) * y
        
        # Store output
        tl.store(out_ptr + base_offset + feature_offsets, out, mask=mask)


def fused_norm_add_mul(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """
    Wrapper function for the fused norm-add-mul operation.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    assert x.shape == y.shape, "x and y must have the same shape."
    
    batch_size, num_features = x.shape
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Prepa
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 28
```
