# KernelBench Level 3 Problem 23: 23_EfficientNetB1.py

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
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Creates a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride of the depthwise convolution.
        :param expand_ratio: Expansion ratio for the hidden layer.
        :return: A sequential MBConv block.
        """
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB1 model.

        :param x: Input tensor, shape (batch_size, 3, 240, 240)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

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

A previous single-shot attempt failed. Error: `failed to translate module to LLVM IR`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    n_channels,
    n_elements_per_channel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    channel_id = pid % n_channels
    element_id = pid // n_channels
    
    channel_start = channel_id * n_elements_per_channel
    offset = channel_start + element_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < channel_start + n_elements_per_channel
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    running_mean = tl.load(running_mean_ptr + channel_id)
    running_var = tl.load(running_var_ptr + channel_id)
    
    normalized = (x - running_mean) / tl.sqrt(running_var + eps)
    out = normalized * weight + bias
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def fused_bn_relu6_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    n_channels,
    n_elements_per_channel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    channel_id = pid % n_channels
    element_id = pid // n_channels
    
    channel_start = channel_id * n_elements_per_channel
    offset = channel_start + element_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < channel_start + n_elements_per_channel
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    running_mean = tl.load(running_mean_ptr + channel_id)
    running_var = tl.load(running_var_ptr + channel_id)
    
    normalized = (x - running_mean) / tl.sqrt(running_var + eps)
    out = normalized * weight + bias
    out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offset, out, mask=mask)


@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_h,
    in_w,
    out_h,
    out_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (channels * out_h * out_w)
    remainder = pid % (channels * out_h * out_w)
    c = remainder // (out_h * out_w)
    remainder = remainder % (out_h * out_w)
    oh = remainder // out_w
    ow = remainder % out_w
    
    ih_start = (oh * in_h) // out_h
    ih_end = ((oh + 1) * in_h + out_h - 1) // out_h
    iw_start = (ow * in_w) // out_w
    iw_end = ((ow + 1) * in_w + out_w - 1) // out_w
    
    acc = 0.0
    count = 0
    
    for ih in range(ih_start, ih_end):
        for iw in range(iw_start, iw_end):
            idx = ((b * channels + c) * in_h + ih) * in_w + iw
            val = tl.load(x_ptr + idx)
            acc += val
            count += 1
    
    out_idx = ((b * channels + c) * out_h + oh) *
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 23
```
