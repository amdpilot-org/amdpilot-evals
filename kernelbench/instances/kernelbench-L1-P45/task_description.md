# KernelBench Level 1 Problem 45: 45_Average_Pooling_2D.py

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
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return self.avg_pool(x)

batch_size = 16
channels = 64
height = 2048
width = 2048
kernel_size = 11

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size]
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

A previous single-shot attempt failed. Error: `at 62:4:
            count += tl.where(valid, 1, 0)

    avg = tl.where(count > 0, acc / count, 0.0)

    output_idx = (
        batch_idx * channels * out_height * out_width +
        channel_idx ...`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    in_height: tl.constexpr,
    in_width: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    batch_idx = pid // (channels * out_height * out_width)
    remainder = pid % (channels * out_height * out_width)
    channel_idx = remainder // (out_height * out_width)
    remainder = remainder % (out_height * out_width)
    out_h = remainder // out_width
    out_w = remainder % out_width
    
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    acc = tl.zeros([1], dtype=tl.float32)
    count = tl.zeros([1], dtype=tl.int32)
    
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            
            valid_h = (in_h >= 0) & (in_h < in_height)
            valid_w = (in_w >= 0) & (in_w < in_width)
            valid = valid_h & valid_w
            
            in_h_clamped = tl.maximum(tl.minimum(in_h, in_height - 1), 0)
            in_w_clamped = tl.maximum(tl.minimum(in_w, in_width - 1), 0)
            
            input_idx = (
                batch_idx * channels * in_height * in_width +
                channel_idx * in_height * in_width +
                in_h_clamped * in_width +
                in_w_clamped
            )
            
            val = tl.load(x_ptr + input_idx)
            acc += tl.where(valid, val, 0.0)
            count += tl.where(valid, 1, 0)
    
    avg = tl.where(count > 0, acc / count, 0.0)
    
    output_idx = (
        batch_idx * channels * out_height * out_width +
        channel_idx * out_height * out_width +
        out_h * out_width +
        out_w
    )
    
    tl.store(out_ptr + output_idx, avg)


def triton_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0
) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, in_height, in_width = x.shape
    
    if stride is None:
        stride = kernel_size
    
    out_height = (in_height + 2 * padding - kernel_size) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty(
        (batch_size, channels, out_height, out_width),
        dtype=x.dtype,
        device=x.device
    )
    
    n_elements = batch_size * channels * out_height * out_width
    BLOCK_SIZE = 256
    
    grid = lambda meta: (
        (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )
    
    avg_pool2d_kernel[grid](
        x,
        out,
        batch_size,
        channels,
        in_height,
        in_width,
   
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 45
```
