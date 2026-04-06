# KernelBench Level 3 Problem 8: 8_ResNetBasicBlock.py

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
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]
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
def bn_relu_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    n_channels,
    n_elements_per_channel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    channel_id = tl.program_id(0)
    for block_start in range(0, n_elements_per_channel, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_per_channel
        x = tl.load(x_ptr + channel_id * n_elements_per_channel + offsets, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + channel_id)
        var = tl.load(var_ptr + channel_id)
        weight = tl.load(weight_ptr + channel_id)
        bias = tl.load(bias_ptr + channel_id)
        x_norm = (x - mean) / tl.sqrt(var + eps)
        x_scaled = x_norm * weight + bias
        out = tl.maximum(x_scaled, 0.0)
        tl.store(out_ptr + channel_id * n_elements_per_channel + offsets, out, mask=mask)


@triton.jit
def bn_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    n_channels,
    n_elements_per_channel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    channel_id = tl.program_id(0)
    for block_start in range(0, n_elements_per_channel, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_per_channel
        x = tl.load(x_ptr + channel_id * n_elements_per_channel + offsets, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + channel_id)
        var = tl.load(var_ptr + channel_id)
        weight = tl.load(weight_ptr + channel_id)
        bias = tl.load(bias_ptr + channel_id)
        x_norm = (x - mean) / tl.sqrt(var + eps)
        out = x_norm * weight + bias
        tl.store(out_ptr + channel_id * n_elements_per_channel + offsets, out, mask=mask)


@triton.jit
def residual_add_relu_kernel(
    x_ptr,
    identity_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    identity = tl.load(identity_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x + identity, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_bn_relu(x: torch.Tensor, bn: nn.BatchNorm2d):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, n_channels, height, width = x.shape
    n_elements_per_channel = batch_size * height * width
    eps = bn.eps
    mean = bn.running_mean.contiguous()
    var = bn.running_var.contiguous()
    weight = bn.weight.contiguous()
    bias = bn.bias.contiguous()
    BLOCK_SIZE = 1024
    grid = (n_channels,)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
        n_channels, n_elements_per_channel, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
   
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 8
```
