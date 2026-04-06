# KernelBench Level 3 Problem 16: 16_DenseNet201.py

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

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(DenseBlock, self).__init__()
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

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        """
        :param growth_rate: The growth rate of the DenseNet (new features added per layer)
        :param num_classes: The number of output classes for classification
        """
        super(Model, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 48, 32]  # Corresponding layers in DenseNet201

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Testing the DenseNet201 model
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [torch.rand(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]

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

A previous single-shot attempt failed. Error: `at 53:24:
                    w_in_mask = (w_in >= 0) & (w_in < width)
                    valid_mask = h_in_mask[:, None] & w_in_mask[None, :]

                    h_in_clamped = tl.maximum(0, tl....`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bn_relu_conv2d_3x3_kernel(
    x_ptr, weight_ptr, bias_ptr, bn_weight_ptr, bn_bias_ptr, bn_mean_ptr, bn_var_ptr,
    out_ptr,
    batch_size, in_channels, out_channels, height, width,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wh, stride_ww,
    stride_outb, stride_outc, stride_outh, stride_outw,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    h_start = pid_h * BLOCK_SIZE_H
    w_start = 0
    
    oc_offset = pid_oc * BLOCK_SIZE_C
    oc_range = oc_offset + tl.arange(0, BLOCK_SIZE_C)
    oc_mask = oc_range < out_channels
    
    h_range = h_start + tl.arange(0, BLOCK_SIZE_H)
    h_mask = h_range < height
    
    for w_block in range((width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W):
        w_range = w_start + w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
        w_mask = w_range < width
        
        mask_2d = oc_mask[:, None] & h_mask[None, :] & w_mask[None, None, :]
        
        acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        
        for ic in range(in_channels):
            for kh in range(3):
                for kw in range(3):
                    h_in = h_range[:, None] + kh - 1
                    w_in = w_range[None, :] + kw - 1
                    
                    h_in_mask = (h_in >= 0) & (h_in < height)
                    w_in_mask = (w_in >= 0) & (w_in < width)
                    valid_mask = h_in_mask[:, None] & w_in_mask[None, :]
                    
                    h_in_clamped = tl.maximum(0, tl.minimum(height - 1, h_in))
                    w_in_clamped = tl.maximum(0, tl.minimum(width - 1, w_in))
                    
                    x_idx = (pid_b * stride_xb + ic * stride_xc + 
                            h_in_clamped * stride_xh + w_in_clamped * stride_xw)
                    x = tl.load(x_ptr + x_idx, mask=valid_mask[None, :, :], other=0.0)
                    
                    w_idx = pid_oc * stride_woc + ic * stride_wic + kh * stride_wh + kw * stride_ww
                    w = tl.load(weight_ptr + w_idx, mask=oc_mask, other=0.0)
                    
                    acc += x * w[:, None, None]
        
        bn_mean = tl.load(bn_mean_ptr + pid_oc, mask=oc_mask, other=0.0)
        bn_var = tl.load(bn_var_ptr + pid_oc, mask=oc_mask, other=0.0)
        bn_weight = tl.load(bn_weight_ptr + pid_oc, mask=oc_mask, other=1.0)
        bn_bias = tl.load(bn_bias_ptr + pid_oc, mask=oc_mask, other=0.0)
        
        norm = (acc - bn_mean[:, None, None]) / tl.sqrt(bn_var[:, None, None] + eps)
        out = bn_weight[:, None, None] * norm + bn_bias[:, None, None]
        out = tl.maximum(out, 0.0)
        
        if bias_ptr is
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 16
```
