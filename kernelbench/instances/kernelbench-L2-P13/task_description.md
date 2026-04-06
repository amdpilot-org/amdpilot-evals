# KernelBench Level 2 Problem 13: 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling.py

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
    Model that performs a series of operations:
    1. Transposed 3D convolution
    2. Mean pooling (across depth)
    3. Addition
    4. Softmax (across channels)
    5. Tanh activation
    6. Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))  # Broadcastable bias over channels
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)                            # (B, C, D, H, W)
        x = x.mean(dim=2, keepdim=True)                       # Mean pool over depth dim (D)
        x = x + self.bias                                     # Bias add per channel
        x = torch.softmax(x, dim=1)                           # Softmax over channels
        x = torch.tanh(x)                                     # Nonlinearity
        x = x * self.scaling_factor                           # Scaling
        return x

# === Test config ===
batch_size = 16
in_channels  = 16  
out_channels = 64  
depth = 32; height = width = 128  
kernel_size  = 3
stride       = 1  
padding = 1
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]

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

A previous single-shot attempt failed. Error: `at 82:19:
        mean_val = mean_val / depth

        # Add bias
        bias_val = tl.load(bias_ptr + c)
        val = mean_val + bias_val

        # Softmax
        exp_val = tl.exp(val - max_va...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    scaling_factor,
):
    # Each program handles one (b, h, w) position
    pid = tl.program_id(0)
    
    # Decode position
    w = pid % width
    pid = pid // width
    h = pid % height
    b = pid // height
    
    # First pass: compute mean across depth for all channels and find max
    max_val = -1e10
    
    for c in range(channels):
        # Compute mean across depth
        mean_val = 0.0
        for d in range(depth):
            x_idx = ((b * channels + c) * depth + d) * (height * width) + h * width + w
            x_val = tl.load(x_ptr + x_idx)
            mean_val += x_val
        
        mean_val = mean_val / depth
        
        # Add bias
        bias_val = tl.load(bias_ptr + c)
        val = mean_val + bias_val
        
        # Track max for softmax
        max_val = tl.maximum(max_val, val)
    
    # Second pass: compute sum of exp for softmax
    exp_sum = 0.0
    
    for c in range(channels):
        # Compute mean across depth
        mean_val = 0.0
        for d in range(depth):
            x_idx = ((b * channels + c) * depth + d) * (height * width) + h * width + w
            x_val = tl.load(x_ptr + x_idx)
            mean_val += x_val
        
        mean_val = mean_val / depth
        
        # Add bias
        bias_val = tl.load(bias_ptr + c)
        val = mean_val + bias_val
        
        # Softmax (numerically stable)
        exp_val = tl.exp(val - max_val)
        exp_sum += exp_val
    
    # Third pass: apply softmax, tanh, and scale
    for c in range(channels):
        # Compute mean across depth
        mean_val = 0.0
        for d in range(depth):
            x_idx = ((b * channels + c) * depth + d) * (height * width) + h * width + w
            x_val = tl.load(x_ptr + x_idx)
            mean_val += x_val
        
        mean_val = mean_val / depth
        
        # Add bias
        bias_val = tl.load(bias_ptr + c)
        val = mean_val + bias_val
        
        # Softmax
        exp_val = tl.exp(val - max_val)
        softmax_val = exp_val / exp_sum
        
        # Tanh
        tanh_val = tl.libdevice.tanh(softmax_val)
        
        # Scale
        out_val = tanh_val * scaling_factor
        
        # Store (only at depth=0 since we pooled)
        out_idx = ((b * channels + c) * 1 + 0) * (height * width) + h * width + w
        tl.store(out_ptr + out_idx, out_val)


def fused_post_conv(x, bias, scaling_factor):
    """
    Fused kernel for: mean pooling + bias add + softmax + tanh + scaling
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    bias = bias.contiguous()
    
    batch_size, channels, depth, height, width = x.shape
    
    # Output shape: (B, C, 1, H, W) after mean pooling
    out = torch.em
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 13
```
