# KernelBench Level 1 Problem 67: 67_conv_standard_1D.py

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
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d(x)

# Test code
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
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

A previous single-shot attempt failed. Error: `at 60:20:
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    # Loop over input channels in blocks
    for ic_block_start in range(0, channels_pe...`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    # Pointers
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    # Dimensions
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    has_bias: tl.constexpr,
    # Strides for input (N, C, L)
    stride_xn, stride_xc, stride_xl,
    # Strides for weight (OC, IC, K)
    stride_woc, stride_wic, stride_wk,
    # Strides for output (N, OC, OL)
    stride_on, stride_ooc, stride_ol,
    # Block sizes
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_OL: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)  # batch
    pid_oc_block = tl.program_id(1)  # output channel block
    pid_ol_block = tl.program_id(2)  # output length block
    
    # Output channel offsets
    oc_start = pid_oc_block * BLOCK_SIZE_OC
    oc_offs = oc_start + tl.arange(0, BLOCK_SIZE_OC)
    oc_mask = oc_offs < out_channels
    
    # Output length offsets
    ol_start = pid_ol_block * BLOCK_SIZE_OL
    ol_offs = ol_start + tl.arange(0, BLOCK_SIZE_OL)
    ol_mask = ol_offs < output_length
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_OL), dtype=tl.float32)
    
    # Calculate group for each output channel
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Loop over input channels in blocks
    for ic_block_start in range(0, channels_per_group, BLOCK_SIZE_IC):
        ic_offs = ic_block_start + tl.arange(0, BLOCK_SIZE_IC)
        ic_mask = ic_offs < channels_per_group
        
        # Calculate actual input channel indices (considering groups)
        # For grouped convolution, each group of output channels connects to corresponding input channels
        group_id = oc_offs // out_channels_per_group
        ic_actual = group_id * channels_per_group + ic_offs
        ic_valid_mask = ic_mask & (ic_actual < in_channels)
        
        # Load weights for this block: (BLOCK_SIZE_OC, BLOCK_SIZE_IC, kernel_size)
        # Weight shape: (out_channels, in_channels//groups, kernel_size)
        w_offs_oc = oc_offs[:, None, None] * stride_woc
        w_offs_ic = ic_offs[None, :, None] * stride_wic
        w_offs_base = pid_oc_block * BLOCK_SIZE_OC * stride_woc + ic_block_start * stride_wic
        
        # Loop over kernel positions
        for k_idx in range(kernel_size):
            # Calculate input positions for this kernel offset
            # input_pos = ol * stride - padding + k * dilation
            il_offs = ol_offs[None, :] * stride - padding + k_idx * dilation
            il_mask = (il_offs >= 0) & (il_offs < input_length)
            
            # Load input values: (BLOCK_SIZE_IC, BLOCK_SIZE_OL)
            x_offs_n = pid_n * stride_xn
            x_offs_ic = ic_actual[:, None] * stride_xc
            x_offs_l = il_offs
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 67
```
