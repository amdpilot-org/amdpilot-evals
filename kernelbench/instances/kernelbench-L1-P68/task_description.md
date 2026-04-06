# KernelBench Level 1 Problem 68: 68_conv_transposed_3D__square_input__asymmetric_kernel.py

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
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv_transpose3d(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.rand(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization
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

A previous single-shot attempt failed. Error: `at 26:12:
    stride_in_n, stride_in_c, stride_in_d, stride_in_w, stride_in_h,
    stride_w_oc, stride_w_ic, stride_w_d, stride_w_w, stride_w_h,
    stride_out_n, stride_out_c, stride_out_d, stride...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C_in, C_out,
    D_in, W_in, H_in,
    D_out, W_out, H_out,
    K_D, K_W, K_H,
    S_D, S_W, S_H,
    P_D, P_W, P_H,
    OP_D, OP_W, OP_H,
    groups,
    has_bias: tl.constexpr,
    stride_in_n, stride_in_c, stride_in_d, stride_in_w, stride_in_h,
    stride_w_oc, stride_w_ic, stride_w_d, stride_w_w, stride_w_h,
    stride_out_n, stride_out_c, stride_out_d, stride_out_w, stride_out_h,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_c_block = tl.program_id(4)

    c_out_start = pid_c_block * BLOCK_C_OUT
    c_out_offsets = c_out_start + tl.arange(0, BLOCK_C_OUT)
    c_out_mask = c_out_offsets < C_out

    d_out = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    w_out = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_out = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    d_mask = d_out < D_out
    w_mask = w_out < W_out
    h_mask = h_out < H_out

    acc = tl.zeros((BLOCK_C_OUT, BLOCK_D, BLOCK_W, BLOCK_H), dtype=tl.float32)

    for kd in range(K_D):
        for kw in range(K_W):
            for kh in range(K_H):
                d_in_raw = d_out * S_D + kd - P_D
                w_in_raw = w_out * S_W + kw - P_W
                h_in_raw = h_out * S_H + kh - P_H

                d_in_valid = (d_in_raw >= 0) & (d_in_raw < D_in)
                w_in_valid = (w_in_raw >= 0) & (w_in_raw < W_in)
                h_in_valid = (h_in_raw >= 0) & (h_in_raw < H_in)

                valid_mask = d_in_valid[:, None, None] & w_in_valid[None, :, None] & h_in_valid[None, None, :]

                d_in = d_in_raw
                w_in = w_in_raw
                h_in = h_in_raw

                for c_in in range(C_in):
                    c_in_group = c_in // (C_in // groups)
                    c_out_group_start = c_in_group * (C_out // groups)
                    c_out_group_end = (c_in_group + 1) * (C_out // groups)

                    c_in_mask = (c_out_offsets >= c_out_group_start) & (c_out_offsets < c_out_group_end)

                    in_offset = (pid_n * stride_in_n +
                                 c_in * stride_in_c +
                                 d_in * stride_in_d +
                                 w_in * stride_in_w +
                                 h_in * stride_in_h)
                    in_val = tl.load(input_ptr + in_offset, mask=d_in_valid[:, None, None] & w_in_valid[None, :, None] & h_in_valid[None, None, :], other=0.0)

                    for oc_idx in range(BLOCK_C_OUT):
                        c_out = c_out_start + oc_idx
                        if c_out >= C_out:
                            continue
                        if not (c_out >= c_out
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 68
```
