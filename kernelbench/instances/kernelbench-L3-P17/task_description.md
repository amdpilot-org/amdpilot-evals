# KernelBench Level 3 Problem 17: 17_SqueezeNetFireModule.py

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
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(Model, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

# Test code
batch_size = 128
num_input_features = 3
num_output_features = 64
height, width = 256, 256
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
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
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val

    acc = tl.maximum(acc, 0.0)

    output_offset = (
        batch_idx * stride_output_batch +
        out_...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_1x1_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_input_batch: tl.constexpr,
    stride_input_channel: tl.constexpr,
    stride_input_height: tl.constexpr,
    stride_input_width: tl.constexpr,
    stride_weight_out: tl.constexpr,
    stride_weight_in: tl.constexpr,
    stride_output_batch: tl.constexpr,
    stride_output_channel: tl.constexpr,
    stride_output_height: tl.constexpr,
    stride_output_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (out_channels * height * width)
    remaining = pid % (out_channels * height * width)
    out_channel_idx = remaining // (height * width)
    remaining = remaining % (height * width)
    h_idx = remaining // width
    w_idx = remaining % width

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for ic in range(0, in_channels, BLOCK_SIZE):
        channel_offsets = ic + tl.arange(0, BLOCK_SIZE)
        mask = channel_offsets < in_channels
        
        input_offset = (
            batch_idx * stride_input_batch +
            channel_offsets * stride_input_channel +
            h_idx * stride_input_height +
            w_idx * stride_input_width
        )
        input_vals = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        
        weight_offset = out_channel_idx * stride_weight_out + channel_offsets * stride_weight_in
        weight_vals = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
        
        acc += tl.sum(input_vals * weight_vals, axis=0)
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    acc = tl.maximum(acc, 0.0)
    
    output_offset = (
        batch_idx * stride_output_batch +
        out_channel_idx * stride_output_channel +
        h_idx * stride_output_height +
        w_idx * stride_output_width
    )
    tl.store(output_ptr + output_offset, acc)


@triton.jit
def conv2d_3x3_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    stride_input_batch: tl.constexpr,
    stride_input_channel: tl.constexpr,
    stride_input_height: tl.constexpr,
    stride_input_width: tl.constexpr,
    stride_weight_out: tl.constexpr,
    stride_weight_in: tl.constexpr,
    stride_weight_h: tl.constexpr,
    stride_weight_w: tl.constexpr,
    stride_output_batch: tl.constexpr,
    stride_output_channel: tl.constexpr,
    stride_output_height: tl.constexpr,
    stride_output_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (out_channels * height * width)
    remaining = pid % (out_channels * height * width)
    out_channel_idx = remaining // (height 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 17
```
