# KernelBench Level 2 Problem 46: 46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py

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
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract1_value
        x = torch.tanh(x)
        x = x - self.subtract2_value
        x = self.avgpool(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
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

A previous single-shot attempt failed. Error: `at 15:8:
    subtract1,
    subtract2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_sub_tanh_sub_kernel(
    x_ptr,
    out_ptr,
    subtract1,
    subtract2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x - subtract1
    x = tl.libdevice.tanh(x)
    x = x - subtract2
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_sub_tanh_sub(x: torch.Tensor, subtract1: float, subtract2: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_sub_tanh_sub_kernel[grid](x, out, subtract1, subtract2, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def avgpool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    kernel_size,
    stride,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_h = tl.cdiv(out_height, BLOCK_SIZE_H)
    num_blocks_w = tl.cdiv(out_width, BLOCK_SIZE_W)
    
    pid_w = pid % num_blocks_w
    pid_h = (pid // num_blocks_w) % num_blocks_h
    pid_n = pid // (num_blocks_h * num_blocks_w)
    
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    
    for c in range(channels):
        for bh in range(BLOCK_SIZE_H):
            for bw in range(BLOCK_SIZE_W):
                out_h = h_start + bh
                out_w = w_start + bw
                
                if out_h < out_height and out_w < out_width:
                    in_h_start = out_h * stride
                    in_w_start = out_w * stride
                    
                    acc = 0.0
                    count = 0
                    
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            in_h = in_h_start + kh
                            in_w = in_w_start + kw
                            
                            if in_h < in_height and in_w < in_width:
                                in_offset = ((pid_n * channels + c) * in_height + in_h) * in_width + in_w
                                val = tl.load(x_ptr + in_offset)
                                acc += val
                                count += 1
                    
                    avg = acc / count
                    out_offset = ((pid_n * channels + c) * out_height + out_h) * out_width + out_w
                    tl.store(out_ptr + out_offset, avg)


def triton_avgpool(x: torch.Tensor, kernel_size: int):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, in_height, i
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 46
```
