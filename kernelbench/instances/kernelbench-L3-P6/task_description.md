# KernelBench Level 3 Problem 6: 6_GoogleNetInceptionModule.py

## Goal

Write an optimized Triton kernel implementation (`ModelNew`) that:
1. Produces the **exact same output** as the PyTorch reference `Model`
2. Is **faster** than the PyTorch baseline
3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)

## PyTorch Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(Model, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]
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

A previous single-shot attempt failed. Error: `at 42:15:
    c = rem // hw
    pos = rem % hw
    h = pos // width
    w = pos % width

    # Max pool over 3x3 window with padding=1
    pooled_val = -1e9
    for dh in range(-1, 2):
        for ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def concat_channels_kernel(
    out_ptr,
    input_ptrs,
    channel_offsets,
    batch_size,
    height,
    width,
    num_branches,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for concatenating multiple 4D tensors along the channel dimension.
    Each program instance handles a block of output elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    hw = height * width
    total_channels = tl.load(channel_offsets + num_branches)
    total_elements = batch_size * total_channels * hw
    mask = offsets < total_elements

    batch_idx = offsets // (total_channels * hw)
    pos_in_batch = offsets % (total_channels * hw)
    channel_idx = pos_in_batch // hw
    spatial_idx = pos_in_batch % hw

    # Determine which branch this channel belongs to
    branch_idx = 0
    for i in range(num_branches):
        start_c = tl.load(channel_offsets + i)
        end_c = tl.load(channel_offsets + i + 1)
        if (channel_idx >= start_c) and (channel_idx < end_c):
            branch_idx = i
            break

    src_channel = channel_idx - tl.load(channel_offsets + branch_idx)
    branch_channels = tl.load(channel_offsets + branch_idx + 1) - tl.load(channel_offsets + branch_idx)
    src_idx = batch_idx * branch_channels * hw + src_channel * hw + spatial_idx

    input_ptr = tl.load(input_ptrs + branch_idx).to(tl.pointer_type(tl.float32))
    val = tl.load(input_ptr + src_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def triton_concat(inputs: list):
    """
    Custom Triton-based concatenation along channel dimension for 4D tensors.
    """
    if len(inputs) == 0:
        return torch.empty(0, device='cuda')
    
    inputs = [x.cuda().contiguous() for x in inputs]
    batch_size = inputs[0].shape[0]
    height = inputs[0].shape[2]
    width = inputs[0].shape[3]
    out_channels = sum(x.shape[1] for x in inputs)
    
    out = torch.empty(batch_size, out_channels, height, width,
                      device=inputs[0].device, dtype=inputs[0].dtype)
    
    input_ptrs = torch.tensor([x.data_ptr() for x in inputs], dtype=torch.int64, device=inputs[0].device)
    channel_counts = [x.shape[1] for x in inputs]
    channel_offsets = [0] + [sum(channel_counts[:i+1]) for i in range(len(channel_counts))]
    channel_offsets_tensor = torch.tensor(channel_offsets, dtype=torch.int32, device=inputs[0].device)
    
    num_branches = len(inputs)
    BLOCK_SIZE = 256
    total_elements = batch_size * out_channels * height * width
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    concat_channels_kernel[grid](
        out, input_ptrs, channel_offsets_tensor,
        batch_size, height, width, num_branches,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


@triton.jit
def fused_pool_conv_kernel(

```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 6
```
