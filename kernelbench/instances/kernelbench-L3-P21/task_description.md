# KernelBench Level 3 Problem 21: 21_EfficientNetMBConv.py

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
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(Model, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x

# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
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

A previous single-shot attempt failed. Error: `at 29:13:
    stride_input_width,
    stride_output_batch,
    stride_output_channel,
    stride_output_height,
    stride_output_width,
    BLOCK_OUT: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
):...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pw_conv_bn_relu6_kernel(
    input_ptr,
    weight_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    eps,
    stride_input_batch,
    stride_input_channel,
    stride_input_height,
    stride_input_width,
    stride_output_batch,
    stride_output_channel,
    stride_output_height,
    stride_output_width,
    BLOCK_OUT: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_oc = tl.program_id(3)

    out_start = pid_oc * BLOCK_OUT
    out_offs = out_start + tl.arange(0, BLOCK_OUT)
    out_mask = out_offs < out_channels

    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

    for ic_block in range(0, in_channels, BLOCK_OUT):
        ic_offs = ic_block + tl.arange(0, BLOCK_OUT)
        ic_mask = ic_offs < in_channels

        in_ptr_off = (pid_b * stride_input_batch +
                      ic_offs * stride_input_channel +
                      pid_h * stride_input_height +
                      pid_w * stride_input_width)
        inp = tl.load(input_ptr + in_ptr_off, mask=ic_mask, other=0.0)

        w_ptr_off = (out_offs[:, None] * out_channels + ic_offs[None, :])
        w_ptr_off = w_ptr_off.ravel()
        w_mask = (out_mask[:, None] & ic_mask[None, :]).ravel()
        weights = tl.load(weight_ptr + w_ptr_off, mask=w_mask, other=0.0)
        weights = weights.reshape(BLOCK_OUT, BLOCK_OUT)

        acc += tl.dot(inp[None, :].to(tl.float32), weights.to(tl.float32))[0, :]

    bn_w = tl.load(bn_weight_ptr + out_offs, mask=out_mask, other=1.0)
    bn_b = tl.load(bn_bias_ptr + out_offs, mask=out_mask, other=0.0)
    bn_m = tl.load(bn_mean_ptr + out_offs, mask=out_mask, other=0.0)
    bn_v = tl.load(bn_var_ptr + out_offs, mask=out_mask, other=1.0)

    acc = (acc - bn_m) / tl.sqrt(bn_v + eps)
    acc = acc * bn_w + bn_b
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 6.0)

    out_ptr_off = (pid_b * stride_output_batch +
                   out_offs * stride_output_channel +
                   pid_h * stride_output_height +
                   pid_w * stride_output_width)
    tl.store(output_ptr + out_ptr_off, acc, mask=out_mask)


@triton.jit
def fused_dw_conv_bn_relu6_kernel(
    input_ptr,
    weight_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    eps,
    stride_input_batch,
    stride_input_channel,
    stride_input_height,
    stride_input_width,
    stride_output_batch,
    stride_output_channel,
    stride_output_height,
    stride_output_width,
    BLOCK_CHAN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 21
```
