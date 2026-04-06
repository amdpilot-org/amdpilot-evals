# KernelBench Level 2 Problem 65: 65_Conv2d_AvgPool_Sigmoid_Sum.py

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
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=[1,2,3]) # Sum over all spatial dimensions
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 384, 384
kernel_size = 3
pool_kernel_size = 4

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
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

A previous single-shot attempt failed. Error: `at 83:4:

        # Average pooling
        pool_val = pool_sum / tl.maximum(pool_count, 1.0)

        # Sigmoid: 1 / (1 + exp(-x))
        sigmoid_val = 1.0 / (1.0 + tl.exp(-pool_val))

        # ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_sigmoid_sum_kernel(
    x_ptr,           # Pointer to input (conv output)
    out_ptr,         # Pointer to output (summed values per batch)
    batch_size,      # Number of samples in batch
    channels,        # Number of channels
    height,          # Input height
    width,           # Input width
    pool_size,       # Pooling kernel size
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch item
    batch_idx = tl.program_id(0)
    
    # Calculate the pooled dimensions
    pooled_height = height // pool_size
    pooled_width = width // pool_size
    total_pooled_elements = pooled_height * pooled_width
    
    # Total elements per batch (channels * pooled_h * pooled_w)
    total_elements = channels * total_pooled_elements
    
    # Initialize accumulator for this batch item
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Process elements in blocks
    for block_start in range(0, total_elements, BLOCK_SIZE):
        # Create offsets for this block
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements
        
        # Decode channel and spatial indices from flat offset
        flat_indices = tl.where(mask, offsets, 0)
        
        # channel = flat_idx // (pooled_h * pooled_w)
        # spatial_idx = flat_idx % (pooled_h * pooled_w)
        # pooled_h_idx = spatial_idx // pooled_w
        # pooled_w_idx = spatial_idx % pooled_w
        spatial_per_channel = pooled_height * pooled_width
        channel_idx = flat_indices // spatial_per_channel
        spatial_idx = flat_indices % spatial_per_channel
        pooled_h_idx = spatial_idx // pooled_width
        pooled_w_idx = spatial_idx % pooled_width
        
        # Calculate the region in input to pool over
        h_start = pooled_h_idx * pool_size
        w_start = pooled_w_idx * pool_size
        
        # Accumulate values in the pooling region
        pool_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        pool_count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for ph in range(pool_size):
            for pw in range(pool_size):
                input_h = h_start + ph
                input_w = w_start + pw
                
                # Check bounds
                valid_h = input_h < height
                valid_w = input_w < width
                valid = valid_h & valid_w
                
                # Calculate input offset
                input_offset = batch_idx * channels * height * width + \
                               channel_idx * height * width + \
                               input_h * width + \
                               input_w
                
                # Load value if valid
                val = tl.load(x_ptr + input_offset, mask=mask & valid, other=0.0)
                pool_sum += val
                pool_count += valid.to(
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 65
```
