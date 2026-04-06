# KernelBench Level 1 Problem 36: 36_RMSNorm_.py

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
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Calculate the RMS along the feature dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

        # Normalize the input by dividing by the RMS
        return x / rms

batch_size = 112
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

A previous single-shot attempt failed. The output was incorrect.

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    out_ptr,
    stride_batch,
    stride_feat,
    stride_dim1,
    stride_dim2,
    batch_size,
    dim1,
    dim2,
    features,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, dim1, dim2) position
    pid = tl.program_id(0)
    
    # Decode pid into batch, dim1, dim2 indices
    dim2_idx = pid % dim2
    remaining = pid // dim2
    dim1_idx = remaining % dim1
    batch_idx = remaining // dim1
    
    # Compute base offset for this position (at feature=0)
    base_offset = (batch_idx * stride_batch + 
                   dim1_idx * stride_dim1 + 
                   dim2_idx * stride_dim2)
    
    # First pass: compute sum of squares across features
    sum_sq = 0.0
    for start in range(0, features, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < features
        # Load values with feature stride
        x = tl.load(x_ptr + base_offset + offsets * stride_feat, 
                    mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute RMS
    mean_sq = sum_sq / features
    rms = tl.sqrt(mean_sq + eps)
    
    # Second pass: normalize
    for start in range(0, features, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < features
        x = tl.load(x_ptr + base_offset + offsets * stride_feat, 
                    mask=mask, other=0.0)
        out = x / rms
        tl.store(out_ptr + base_offset + offsets * stride_feat, out, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float):
    """
    Triton-based RMS Normalization
    
    Args:
        x: Input tensor of shape (batch_size, features, dim1, dim2)
        eps: Small value added to denominator for numerical stability
    
    Returns:
        Normalized tensor of same shape as input
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, features, dim1, dim2 = x.shape
    
    # Prepare output tensor
    out = torch.empty_like(x)
    
    # Get strides
    stride_batch = x.stride(0)
    stride_feat = x.stride(1)
    stride_dim1 = x.stride(2)
    stride_dim2 = x.stride(3)
    
    # Total number of positions (batch * dim1 * dim2)
    num_positions = batch_size * dim1 * dim2
    
    # Choose block size based on feature dimension
    BLOCK_SIZE = 64  # Fixed block size, can be tuned
    
    # Launch kernel
    grid = (num_positions,)
    rms_norm_kernel[grid](
        x, out,
        stride_batch, stride_feat, stride_dim1, stride_dim2,
        batch_size, dim1, dim2,
        features, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 36
```
