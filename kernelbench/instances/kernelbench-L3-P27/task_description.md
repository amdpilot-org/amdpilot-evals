# KernelBench Level 3 Problem 27: 27_RegNet.py

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
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(Model, self).__init__()

        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        # Construct the stages with their respective blocks
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)
    
    def _make_stage(self, in_channels, out_channels):
        """
        Creates a simple block for each stage.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return: nn.Sequential block with convolutional layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass through the RegNet model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        x = self.fc(x)
        return x

# Test code for the RegNet model
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor of shape (batch_size, input_channels, height, width) """
    return [torch.rand(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    """ Initializes model parameters """
    return [input_channels, stages, block_widths, output_classes]
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

A previous single-shot attempt failed. Error: `at 20:15:
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    spatial_elements = height * width
    acc = 0.0

    for block_start in range(0, spatial_elements, BLOCK_SIZE):
  ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_relu_kernel(
    x_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_running_mean_ptr,
    bn_running_var_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    channel_id = tl.program_id(0)
    elements_per_channel = batch_size * height * width
    
    bn_weight = tl.load(bn_weight_ptr + channel_id)
    bn_bias = tl.load(bn_bias_ptr + channel_id)
    bn_mean = tl.load(bn_running_mean_ptr + channel_id)
    bn_var = tl.load(bn_running_var_ptr + channel_id)
    
    for block_start in range(0, elements_per_channel, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_channel
        
        x = tl.load(x_ptr + channel_id * elements_per_channel + offsets, mask=mask, other=0.0)
        normalized = (x - bn_mean) / tl.sqrt(bn_var + eps) * bn_weight + bn_bias
        activated = tl.maximum(normalized, 0.0)
        tl.store(out_ptr + channel_id * elements_per_channel + offsets, activated, mask=mask)


def fused_bn_relu(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, eps=1e-5):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (channels,)
    fused_bn_relu_kernel[grid](
        x, bn_weight, bn_bias, bn_running_mean, bn_running_var, out,
        batch_size, channels, height, width, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    spatial_elements = height * width
    acc = 0.0
    
    for block_start in range(0, spatial_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < spatial_elements
        base_offset = (batch_id * channels + channel_id) * spatial_elements
        x = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        acc += tl.sum(x, mask=mask)
    
    mean = acc / spatial_elements
    out_offset = batch_id * channels + channel_id
    tl.store(out_ptr + out_offset, mean)


def global_avg_pool(x):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    out = torch.empty((batch_size, channels), dtype=x.dtype, device=x.device)
    BLOCK_SIZE = 256
    grid = (batch_size, channels)
    global_avg_pool_kernel[grid](
        x, out, batch_size, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the f
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 27
```
