# KernelBench Level 3 Problem 25: 25_ShuffleNetUnit.py

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
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(Model, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose
        x = x.transpose(1, 2).contiguous()
        
        # Flatten
        x = x.view(batch_size, -1, height, width)
        
        return x
    
batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [input_channels, out_channels, groups]
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
def channel_shuffle_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    
    if pid >= total_elements:
        return
    
    tmp = pid
    w = tmp % width
    tmp = tmp // width
    h = tmp % height
    tmp = tmp // height
    c = tmp % channels
    b = tmp // channels
    
    src_c = (c // groups) * channels_per_group + (c % groups)
    src_idx = b * channels * height * width + src_c * height * width + h * width + w
    
    val = tl.load(x_ptr + src_idx)
    tl.store(out_ptr + pid, val)


def triton_channel_shuffle(x, groups):
    batch_size, channels, height, width = x.shape
    channels_per_group = channels // groups
    
    x = x.contiguous()
    out = torch.empty_like(x)
    
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 256
    
    grid = lambda meta: ((total_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    channel_shuffle_kernel[grid](
        x, out, batch_size, channels, height, width,
        groups, channels_per_group,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


@triton.jit
def fused_conv1x1_bn_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_running_mean_ptr,
    bn_running_var_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch_size * out_channels * height * width
    
    if pid >= total_elements:
        return
    
    tmp = pid
    w = tmp % width
    tmp = tmp // width
    h = tmp % height
    tmp = tmp // height
    oc = tmp % out_channels
    n = tmp // out_channels
    
    group_id = oc // (out_channels // groups)
    group_start_ic = group_id * (in_channels // groups)
    group_size_ic = in_channels // groups
    
    acc = 0.0
    for ic_offset in range(group_size_ic):
        ic = group_start_ic + ic_offset
        weight_idx = oc * group_size_ic + ic_offset
        x_idx = n * in_channels * height * width + ic * height * width + h * width + w
        
        x_val = tl.load(x_ptr + x_idx)
        w_val = tl.load(weight_ptr + weight_idx)
        acc += x_val * w_val
    
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + oc)
    
    if bn_running_mean_ptr is not None:
        mean = tl.load(bn_running_mean_ptr + oc)
        var = tl.load(bn_running_var_ptr + oc)
        bn_w = tl.load(bn_weight_ptr + oc) if bn_weight_ptr is not None else 1.0
        bn_b = tl.load(bn_bias_ptr + oc) if bn_bias_ptr is not None else 0.0
        
        acc = (acc - mean) / tl.sqrt(var + eps) * bn_w + bn_b
    
    acc = tl.maximum(acc, 0.0)
    tl.store(o
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 25
```
