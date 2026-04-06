# KernelBench Level 1 Problem 87: 87_conv_pointwise_2D.py

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
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.conv1d(x)

# Test code
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]
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

A previous single-shot attempt failed. Error: `at 58:15:

        # Load weight values: (C_out, C_in)
        # Index: c_out * C_in + c_in
        weight_offsets = (
            out_offsets[:, None] * in_channels +
            in_offsets[None, ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # Each program handles one spatial position (n, h, w) and a block of output channels
    pid_spatial = tl.program_id(0)
    pid_out = tl.program_id(1)
    
    # Decode spatial position (n, h, w)
    n = pid_spatial // (height * width)
    hw = pid_spatial % (height * width)
    h = hw // width
    w = hw % width
    
    # Output channel block
    out_block_start = pid_out * BLOCK_OUT
    out_offsets = out_block_start + tl.arange(0, BLOCK_OUT)
    out_mask = out_offsets < out_channels
    
    # Accumulator for output
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    
    # Loop over input channels in blocks
    for in_block_start in range(0, in_channels, BLOCK_IN):
        in_offsets = in_block_start + tl.arange(0, BLOCK_IN)
        in_mask = in_offsets < in_channels
        
        # Load input values: (N, C_in, H, W)
        # Index: n * C_in * H * W + c_in * H * W + h * W + w
        input_offsets = (
            n * in_channels * height * width +
            in_offsets * height * width +
            h * width +
            w
        )
        input_vals = tl.load(input_ptr + input_offsets, mask=in_mask, other=0.0)
        
        # Load weight values: (C_out, C_in)
        # Index: c_out * C_in + c_in
        weight_offsets = (
            out_offsets[:, None] * in_channels +
            in_offsets[None, :]
        )
        weight_mask = out_mask[:, None] & in_mask[None, :]
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        
        # Matrix multiply: acc (BLOCK_OUT,) += weight (BLOCK_OUT, BLOCK_IN) @ input (BLOCK_IN,)
        acc += tl.dot(weight_vals, input_vals[:, None])[:, 0]
    
    # Add bias if present
    if has_bias:
        bias_vals = tl.load(bias_ptr + out_offsets, mask=out_mask, other=0.0)
        acc += bias_vals
    
    # Store output: (N, C_out, H, W)
    output_offsets = (
        n * out_channels * height * width +
        out_offsets * height * width +
        h * width +
        w
    )
    tl.store(output_ptr + output_offsets, acc, mask=out_mask)


def triton_pointwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    BLOCK_IN: int = 32,
    BLOCK_OUT: int = 32,
) -> torch.Tensor:
    """
    Triton-based pointwise convolution (1x1 conv).
    
    Args:
        x: Input tensor of shape (N, C_in, H, W)
        weight: Weight tensor of shape (C_out, C_in)
        bias: Optional bias tensor of shape (C_out,)
        BLOCK_IN: Block size for input channels
        BLOCK_OUT: Block size for out
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 87
```
