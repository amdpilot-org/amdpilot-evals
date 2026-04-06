# KernelBench Level 2 Problem 7: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd.py

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
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x

batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
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

A previous single-shot attempt failed. Error: `at 46:17:

    # Apply LeakyReLU: x if x > 0 else 0.01 * x
    # Since we already applied ReLU, x >= 0, so LeakyReLU is just x
    # But to be correct for the full sequence, we apply it properly:
 ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_activation_bias_kernel(
    x_ptr,           # Pointer to input (conv output)
    bias_ptr,        # Pointer to bias
    out_ptr,         # Pointer to output
    N: tl.constexpr, # Batch size
    C: tl.constexpr, # Channels
    D: tl.constexpr, # Depth
    H: tl.constexpr, # Height
    W: tl.constexpr, # Width
    stride_n: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Total number of elements
    total_elements = N * C * D * H * W
    
    # Starting index for this program
    block_start = pid * BLOCK_SIZE
    
    # Create offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(0, x)
    x = tl.maximum(x, 0.0)
    
    # Apply LeakyReLU: x if x > 0 else 0.01 * x
    # Since we already applied ReLU, x >= 0, so LeakyReLU is just x
    # But to be correct for the full sequence, we apply it properly:
    leaky_slope = 0.01
    x = tl.where(x > 0, x, leaky_slope * x)
    
    # Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_inner = tl.libdevice.tanh(inner)
    x = 0.5 * x * (1.0 + tanh_inner)
    
    # Apply Sigmoid: 1 / (1 + exp(-x))
    x = tl.libdevice.exp(-x)
    x = 1.0 / (1.0 + x)
    
    # Now we need to add bias - need to compute channel index for each element
    # Reshape offset to get channel index
    # offset = n*C*D*H*W + c*D*H*W + d*H*W + h*W + w
    # c = (offset % (C*D*H*W)) // (D*H*W)
    elements_per_channel = D * H * W
    channel_indices = (offsets % (C * elements_per_channel)) // elements_per_channel
    
    # Load bias values (broadcasted per channel)
    bias = tl.load(bias_ptr + channel_indices, mask=mask, other=0.0)
    
    # Add bias
    x = x + bias
    
    # Store output
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor):
    """
    Fuses ReLU, LeakyReLU, GELU, Sigmoid, and bias addition into a single Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert bias.is_cuda, "Bias tensor must be on CUDA."
    
    x = x.contiguous()
    bias = bias.contiguous()
    
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 256  # Tunable parameter
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_activation_bias_kernel[grid](
        x,
        bias,
        out,
        N, C, D, H, W,
        x.stride(0),
        x.stride(1),
       
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 7
```
