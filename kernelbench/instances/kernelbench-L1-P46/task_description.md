# KernelBench Level 1 Problem 46: 46_Average_Pooling_3D.py

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
    Simple model that performs 3D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return self.avg_pool(x)

batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
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
def avg_pool3d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    in_depth,
    in_height,
    in_width,
    out_depth,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_out = batch_size * channels * out_depth * out_height * out_width
    mask = pos < total_out

    # Decode flat position to 5D indices
    tmp = pos
    out_w_idx = tmp % out_width
    tmp = tmp // out_width
    out_h_idx = tmp % out_height
    tmp = tmp // out_height
    out_d_idx = tmp % out_depth
    tmp = tmp // out_depth
    c = tmp % channels
    n = tmp // channels

    # Input window start
    in_d_start = out_d_idx * stride - padding
    in_h_start = out_h_idx * stride - padding
    in_w_start = out_w_idx * stride - padding

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                in_d = in_d_start + kd
                in_h = in_h_start + kh
                in_w = in_w_start + kw

                valid_d = (in_d >= 0) & (in_d < in_depth)
                valid_h = (in_h >= 0) & (in_h < in_height)
                valid_w = (in_w >= 0) & (in_w < in_width)
                valid = valid_d & valid_h & valid_w

                input_pos = (
                    (((n * channels + c) * in_depth + in_d) * in_height + in_h) * in_width + in_w
                )

                val = tl.load(input_ptr + input_pos, mask=mask & valid, other=0.0)
                acc += val
                count += valid.to(tl.float32)

    avg = acc / tl.maximum(count, 1.0)
    tl.store(output_ptr + pos, avg, mask=mask)


def triton_avg_pool3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    batch_size, channels, in_depth, in_height, in_width = x.shape
    out_depth = (in_depth + 2 * padding - kernel_size) // stride + 1
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty(
        (batch_size, channels, out_depth, out_height, out_width),
        dtype=x.dtype,
        device=x.device,
    )

    total_out = batch_size * channels * out_depth * out_height * out_width
    BLOCK_SIZE = 256
    grid = ((total_out + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    avg_pool3d_kernel[grid](
        x,
        out,
        batch_size,
        channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    r
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 46
```
