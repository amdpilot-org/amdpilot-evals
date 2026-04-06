# KernelBench Level 3 Problem 20: 20_MobileNetV2.py

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
        MobileNetV2 architecture implementation in PyTorch.

        :param num_classes: The number of output classes. Default is 1000.
        """
        super(Model, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            """
            Inverted Residual Block for MobileNetV2.
            """
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)[0])
                input_channel = output_channel

        # Building last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))

        # Final layer
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the MobileNetV2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

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

A previous single-shot attempt failed. Error: `at 41:19:
            offs_k = start_k + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < in_features

            x = tl.load(x_ptr + pid_m * in_features + offs_k, mask=mask_k, other=0.0)....`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_relu6_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    stride,
    n_channels,
    n_elements_per_channel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    channel_id = tl.program_id(0)
    
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    running_mean = tl.load(running_mean_ptr + channel_id)
    running_var = tl.load(running_var_ptr + channel_id)
    
    for block_start in range(0, n_elements_per_channel, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_per_channel
        
        x = tl.load(x_ptr + channel_id * stride + offsets, mask=mask, other=0.0)
        
        x_norm = (x - running_mean) * tl.rsqrt(running_var + eps)
        x_scaled = x_norm * weight + bias
        
        x_out = tl.minimum(tl.maximum(x_scaled, 0.0), 6.0)
        
        tl.store(out_ptr + channel_id * stride + offsets, x_out, mask=mask)


@triton.jit
def fused_bn_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    stride,
    n_channels,
    n_elements_per_channel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    channel_id = tl.program_id(0)
    
    weight = tl.load(weight_ptr + channel_id)
    bias = tl.load(bias_ptr + channel_id)
    running_mean = tl.load(running_mean_ptr + channel_id)
    running_var = tl.load(running_var_ptr + channel_id)
    
    for block_start in range(0, n_elements_per_channel, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_per_channel
        
        x = tl.load(x_ptr + channel_id * stride + offsets, mask=mask, other=0.0)
        
        x_norm = (x - running_mean) * tl.rsqrt(running_var + eps)
        x_out = x_norm * weight + bias
        
        tl.store(out_ptr + channel_id * stride + offsets, x_out, mask=mask)


def fused_bn_relu6(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, eps=1e-5):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    
    batch_size, n_channels, height, width = x.shape
    stride = batch_size * height * width
    n_elements_per_channel = stride
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 256
    grid = (n_channels,)
    
    fused_bn_relu6_kernel[grid](
        x, bn_weight, bn_bias, bn_running_mean, bn_running_var, out,
        stride, n_channels, n_elements_per_channel, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out


def fused_bn(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, eps=1e-5):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    
    batch_size, n_channels, height, width = x.shape
    stride = batch_size * height * width
    n_elements_per_channel = stride
    
    out = torch.empty_like(x)
    
    BLOCK_SIZ
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 20
```
