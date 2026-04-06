# KernelBench Level 3 Problem 14: 14_DenseNet121DenseBlock.py

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
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(Model, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)  # Concatenate along channel axis
        return x
    
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_layers, num_input_features , growth_rate]
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
def bn_relu_fused_kernel(
    x_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    out_ptr,
    n_elements,
    channels,
    hw_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused BatchNorm + ReLU kernel for NCHW tensor format.
    Each program handles a contiguous block of elements.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Calculate channel for each element: channel = (offset // (H * W)) % C
    channel_id = (offsets // hw_size) % channels

    # Load BatchNorm parameters for the corresponding channel
    weight = tl.load(bn_weight_ptr + channel_id)
    bias = tl.load(bn_bias_ptr + channel_id)
    mean = tl.load(bn_mean_ptr + channel_id)
    var = tl.load(bn_var_ptr + channel_id)

    # Apply BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * weight + bias

    # Apply ReLU
    out = tl.maximum(x_scaled, 0.0)

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, bn_module, eps=1e-5):
    """
    Fused BatchNorm + ReLU operation using Triton.
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    batch_size, channels, height, width = x.shape
    n_elements = x.numel()
    hw_size = height * width

    # Prepare output tensor
    out = torch.empty_like(x)

    # Get BatchNorm parameters
    bn_weight = bn_module.weight
    bn_bias = bn_module.bias
    bn_mean = bn_module.running_mean
    bn_var = bn_module.running_var

    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    bn_relu_fused_kernel[grid](
        x, bn_weight, bn_bias, bn_mean, bn_var, out,
        n_elements, channels, hw_size, eps, BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class FusedLayer(nn.Module):
    """
    Layer with fused BatchNorm + ReLU, followed by Conv2d.
    Dropout is omitted as p=0.0.
    """
    def __init__(self, in_features, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = fused_bn_relu(x, self.bn)
        x = self.conv(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        Optimized Dense Block with fused BatchNorm + ReLU and pre-allocated output.
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(FusedLayer(num_input_features + i * growth_rate, growth_rate))
        self.laye
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 14
```
