# KernelBench Level 1 Problem 60: 60_conv_standard_3D__square_input__asymmetric_kernel.py

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
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return self.conv3d(x)

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, width, height, depth)
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

A previous single-shot attempt failed. Error: `at 29:13:
    # Config
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program IDs
    pid_n = tl.pr...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    # Pointers
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Dimensions
    N, C, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    stride, padding, dilation,
    # Strides input
    stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
    # Strides output
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    # Strides weight
    stride_w_oc, stride_w_ic, stride_w_kd, stride_w_kh, stride_w_kw,
    # Config
    HAS_BIAS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_od = tl.program_id(2)
    pid_oh = tl.program_id(3)
    pid_ow = tl.program_id(4)
    
    # Output block start
    out_d_start = pid_od * BLOCK_D
    out_h_start = pid_oh * BLOCK_H
    out_w_start = pid_ow * BLOCK_W
    
    # Loop over output block
    for d_off in range(BLOCK_D):
        for h_off in range(BLOCK_H):
            for w_off in range(BLOCK_W):
                cur_d = out_d_start + d_off
                cur_h = out_h_start + h_off
                cur_w = out_w_start + w_off
                
                # Bounds check
                if cur_d >= OD or cur_h >= OH or cur_w >= OW:
                    continue
                
                acc = 0.0
                
                # Loop over input channels in blocks
                for c_start in range(0, C, BLOCK_C):
                    c_end = min(c_start + BLOCK_C, C)
                    
                    # Loop over kernel dimensions
                    for kd in range(KD):
                        for kh in range(KH):
                            for kw in range(KW):
                                # Calculate input position
                                in_d = cur_d * stride + kd * dilation - padding
                                in_h = cur_h * stride + kh * dilation - padding
                                in_w = cur_w * stride + kw * dilation - padding
                                
                                # Check input bounds
                                if in_d < 0 or in_d >= D or \
                                   in_h < 0 or in_h >= H or \
                                   in_w < 0 or in_w >= W:
                                    continue
                                
                                # Load and compute for channel block
                                for c_idx in range(c_start, c_end):
                                    in_idx = pid_n * stride_in_n + \
                                            c_idx * stride_in_c + \
                                            in_d * stride_in_d + \
                                            in_h * stride_in_h + \
                                            in_w * stri
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 60
```
