# KernelBench Level 1 Problem 84: 84_conv_depthwise_2D_asymmetric_input_square_kernel.py

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
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# Test code
batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
width_in = 512
height_in = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.rand(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
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

A previous single-shot attempt failed. Error: `at 44:12:
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_height = tl.program_id(2)

    # Output height position for this program
    h_out_start = pid_height * BLOCK_H...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    # Pointers
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    # Dimensions
    batch_size,
    in_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    # Strides
    stride_x_batch,
    stride_x_channel,
    stride_x_height,
    stride_x_width,
    stride_w_channel,
    stride_w_kernel,
    stride_out_batch,
    stride_out_channel,
    stride_out_height,
    stride_out_width,
    # Config
    HAS_BIAS: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_height = tl.program_id(2)
    
    # Output height position for this program
    h_out_start = pid_height * BLOCK_HEIGHT
    
    # Loop over output height block
    for h_block in range(BLOCK_HEIGHT):
        h_out = h_out_start + h_block
        if h_out >= height_out:
            break
        
        # Calculate input height start
        h_in_start = h_out * stride
        
        # Loop over output width
        for w_out in range(width_out):
            # Calculate input width start
            w_in_start = w_out * stride
            
            # Accumulator for convolution
            acc = 0.0
            
            # Kernel loop
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    h_in = h_in_start + kh
                    w_in = w_in_start + kw
                    
                    # Bounds check
                    if (h_in >= 0 and h_in < height_in and 
                        w_in >= 0 and w_in < width_in):
                        # Load input
                        x_offset = (pid_batch * stride_x_batch + 
                                   pid_channel * stride_x_channel + 
                                   h_in * stride_x_height + 
                                   w_in * stride_x_width)
                        x_val = tl.load(x_ptr + x_offset)
                        
                        # Load weight
                        w_offset = (pid_channel * stride_w_channel + 
                                   kh * stride_w_kernel + kw)
                        w_val = tl.load(weight_ptr + w_offset)
                        
                        # Accumulate
                        acc += x_val * w_val
            
            # Add bias if present
            if HAS_BIAS:
                bias_val = tl.load(bias_ptr + pid_channel)
                acc += bias_val
            
            # Store output
            out_offset = (pid_batch * stride_out_batch + 
                         pid_channel * stride_out_channel + 
                         h_out * stride_out_height + 
                         w_out * stride_out_width)
            tl.store(out_ptr + out_offset, acc)


def triton_depthw
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 84
```
