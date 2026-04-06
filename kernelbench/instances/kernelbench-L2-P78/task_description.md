# KernelBench Level 2 Problem 78: 78_ConvTranspose3d_Max_Max_Sum.py

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
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = torch.sum(x, dim=1, keepdim=True) 
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

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

A previous single-shot attempt failed. The output was incorrect.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_sum_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pool1_k: tl.constexpr,
    pool2_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Each program handles one output element (batch, 1, d, h, w)
    pid = tl.program_id(0)
    
    # Decode the output position
    total_w = (width + pool1_k - 1) // pool1_k
    total_w = (total_w + pool2_k - 1) // pool2_k
    total_h = (height + pool1_k - 1) // pool1_k
    total_h = (total_h + pool2_k - 1) // pool2_k
    total_d = (depth + pool1_k - 1) // pool1_k
    total_d = (total_d + pool2_k - 1) // pool2_k
    
    w_idx = pid % total_w
    h_idx = (pid // total_w) % total_h
    d_idx = (pid // (total_w * total_h)) % total_d
    b_idx = pid // (total_w * total_h * total_d)
    
    # Calculate the region in the original tensor after both poolings
    start_w = w_idx * pool1_k * pool2_k
    start_h = h_idx * pool1_k * pool2_k
    start_d = d_idx * pool1_k * pool2_k
    
    # Accumulate max over channels and spatial region
    acc = 0.0
    
    for c in range(channels):
        # Max pool 1
        max_val1 = -1e30
        for pd in range(pool1_k):
            for ph in range(pool1_k):
                for pw in range(pool1_k):
                    orig_d = start_d // pool2_k + pd
                    orig_h = start_h // pool2_k + ph
                    orig_w = start_w // pool2_k + pw
                    if orig_d < depth and orig_h < height and orig_w < width:
                        offset = b_idx * channels * depth * height * width + \
                                 c * depth * height * width + \
                                 orig_d * height * width + \
                                 orig_h * width + \
                                 orig_w
                        val = tl.load(x_ptr + offset)
                        if val > max_val1:
                            max_val1 = val
        
        # Max pool 2 on the result of pool 1 (approximated by larger window)
        max_val2 = -1e30
        for pd in range(pool1_k * pool2_k):
            for ph in range(pool1_k * pool2_k):
                for pw in range(pool1_k * pool2_k):
                    orig_d = start_d + pd
                    orig_h = start_h + ph
                    orig_w = start_w + pw
                    if orig_d < depth and orig_h < height and orig_w < width:
                        offset = b_idx * channels * depth * height * width + \
                                 c * depth * height * width + \
                                 orig_d * height * width + \
                                 orig_h * width + \
                                 orig_w
                        val = tl.load(x_ptr + offset)
    
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 78
```
