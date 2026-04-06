# KernelBench Level 3 Problem 45: 45_UNetSoftmax.py

## Goal

Write an optimized Triton kernel implementation (`ModelNew`) that:
1. Produces the **exact same output** as the PyTorch reference `Model`
2. Is **faster** than the PyTorch baseline
3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)

## PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

# U-Net Implementation
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param features: Number of base features (will be doubled in each layer)
        """
        super(Model, self).__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)
    
batch_size = 8
in_channels = 8
out_channels = 4
height = 64
width = 512
features = 64
# Test code for UNet
def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, features]

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
def fused_bn_softmax_kernel(
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
    """
    Fused BatchNorm + Softmax kernel.
    Applies BatchNorm followed by Softmax along the last dimension (width).
    """
    pid = tl.program_id(0)
    batch_idx = pid // (channels * height)
    remaining = pid % (channels * height)
    channel_idx = remaining // height
    height_idx = remaining % height

    base_offset = batch_idx * channels * height * width + channel_idx * height * width + height_idx * width

    # Load running mean and var for this channel
    mean = tl.load(bn_running_mean_ptr + channel_idx)
    var = tl.load(bn_running_var_ptr + channel_idx)
    weight = tl.load(bn_weight_ptr + channel_idx)
    bias = tl.load(bn_bias_ptr + channel_idx)

    # First pass: compute normalized values and find max for numerical stability
    max_val = -1e9
    for block_start in range(0, width, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        x_vals = tl.load(x_ptr + base_offset + offsets, mask=mask, other=-1e9)
        normalized = (x_vals - mean) / tl.sqrt(var + eps)
        scaled = normalized * weight + bias
        max_val = tl.maximum(max_val, tl.max(scaled, axis=0))

    # Second pass: compute exp and sum
    sum_exp = 0.0
    for block_start in range(0, width, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        x_vals = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        normalized = (x_vals - mean) / tl.sqrt(var + eps)
        scaled = normalized * weight + bias
        exp_vals = tl.exp(scaled - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)

    # Third pass: compute softmax and store
    for block_start in range(0, width, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        x_vals = tl.load(x_ptr + base_offset + offsets, mask=mask, other=0.0)
        normalized = (x_vals - mean) / tl.sqrt(var + eps)
        scaled = normalized * weight + bias
        exp_vals = tl.exp(scaled - max_val)
        softmax_out = exp_vals / (sum_exp + eps)
        tl.store(out_ptr + base_offset + offsets, softmax_out, mask=mask)


@triton.jit
def maxpool2d_kernel(
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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized MaxPool2d kernel.
    """
    pid = tl.program_id(0)
    total_elems = batch_size * channels * out_height * out_width
    if pid >= total_elems:
        return

    remaining = pid
    out_w_idx = remaining % out_width
    remaining = remaining // ou
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 45
```
