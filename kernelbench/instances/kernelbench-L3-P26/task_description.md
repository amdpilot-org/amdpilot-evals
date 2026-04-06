# KernelBench Level 3 Problem 26: 26_ShuffleNet.py

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

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ShuffleNetUnit, self).__init__()
        
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

class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        """
        ShuffleNet architecture.

        :param num_classes: Number of output classes.
        :param groups: Number of groups for group convolution.
        :param stages_repeats: List of ints specifying the number of repeats for each stage.
        :param stages_out_channels: List of ints specifying the output channels for each stage.
        """
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        """
        Helper function to create a stage of ShuffleNet units.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param repeats: Number of ShuffleNet units in the stage.
        :param groups: Number of groups for group convolution.
        :return: nn.Sequential containing the stage.
        """
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet.

        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]
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

A previous single-shot attempt failed. Error: `at 23:12:
    """
    Triton kernel for channel shuffle operation.
    Reshapes, transposes, and flattens channels efficiently.
    """
    pid = tl.program_id(0)
    total_elements = batch_size * ...`

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
    """
    Triton kernel for channel shuffle operation.
    Reshapes, transposes, and flattens channels efficiently.
    """
    pid = tl.program_id(0)
    total_elements = batch_size * channels * height * width
    
    # Each program handles BLOCK_SIZE elements
    for offset in range(BLOCK_SIZE):
        idx = pid * BLOCK_SIZE + offset
        if idx >= total_elements:
            break
        
        # Decode the flat index to (batch, channel, h, w)
        tmp = idx
        w = tmp % width
        tmp = tmp // width
        h = tmp % height
        tmp = tmp // height
        c = tmp % channels
        n = tmp // channels
        
        # Calculate new channel after shuffle
        # Original: (batch, groups, channels_per_group, h, w)
        # Transpose: (batch, channels_per_group, groups, h, w)
        # Flatten: (batch, channels, h, w)
        group_idx = c // channels_per_group
        channel_in_group = c % channels_per_group
        new_c = channel_in_group * groups + group_idx
        
        # Calculate new flat index
        new_idx = n * channels * height * width + new_c * height * width + h * width + w
        
        # Load and store
        val = tl.load(x_ptr + idx)
        tl.store(out_ptr + new_idx, val)


def triton_channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Triton-based channel shuffle operation.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, channels, height, width = x.shape
    channels_per_group = channels // groups
    
    out = torch.empty_like(x)
    
    total_elements = x.numel()
    BLOCK_SIZE = 256
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    channel_shuffle_kernel[grid](
        x, out, batch_size, channels, height, width,
        groups, channels_per_group,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


@triton.jit
def bn_relu_fused_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused BatchNorm + ReLU kernel.
    Applies: ReLU((x - mean) / sqrt(var + eps) * weight + bias)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load BN parameters (assuming they're broadcasted to match x shape)
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    var = tl.load(var_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_p
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 26
```
