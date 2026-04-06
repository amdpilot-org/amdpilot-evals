# KernelBench Level 2 Problem 35: 35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py

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
    Model that performs a convolution, subtracts a value, applies HardSwish, MaxPool, and Mish activation functions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = x - self.subtract_value
        x = torch.nn.functional.hardswish(x)
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]
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

A previous single-shot attempt failed. Error: `at 45:21:
                          in_h * in_width + in_w)
                val = tl.load(x_ptr + in_idx)

                val = val - subtract_value
                val_plus_3 = val + 3.0
        ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    subtract_value,
    pool_kernel_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    out_w = pid % out_width
    out_h = (pid // out_width) % out_height
    out_c = (pid // (out_width * out_height)) % channels
    out_n = pid // (out_width * out_height * channels)
    
    in_h_start = out_h * pool_kernel_size
    in_w_start = out_w * pool_kernel_size
    
    max_val = -1e30
    
    for ph in range(pool_kernel_size):
        for pw in range(pool_kernel_size):
            in_h = in_h_start + ph
            in_w = in_w_start + pw
            
            if in_h < in_height and in_w < in_width:
                in_idx = (out_n * channels * in_height * in_width + 
                          out_c * in_height * in_width + 
                          in_h * in_width + in_w)
                val = tl.load(x_ptr + in_idx)
                
                val = val - subtract_value
                val_plus_3 = val + 3.0
                val_clamped = tl.maximum(tl.minimum(val_plus_3, 6.0), 0.0)
                val = val * val_clamped / 6.0
                
                max_val = tl.maximum(max_val, val)
    
    softplus = tl.log(1.0 + tl.exp(max_val))
    mish = max_val * tl.tanh(softplus)
    
    tl.store(out_ptr + pid, mish)


def fused_post_conv(x: torch.Tensor, subtract_value: float, pool_kernel_size: int):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, in_height, in_width = x.shape
    out_height = in_height // pool_kernel_size
    out_width = in_width // pool_kernel_size
    
    out = torch.empty((batch_size, channels, out_height, out_width), 
                      dtype=x.dtype, device=x.device)
    
    n_elements = batch_size * channels * out_height * out_width
    BLOCK_SIZE = 256
    
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    fused_post_conv_kernel[grid](
        x, out,
        batch_size, channels,
        in_height, in_width,
        out_height, out_width,
        subtract_value, pool_kernel_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then fuses subtract, HardSwish, 
    MaxPool, and Mish into a single Triton kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = fused_pos
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 35
```
