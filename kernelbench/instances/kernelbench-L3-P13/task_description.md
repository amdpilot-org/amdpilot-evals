# KernelBench Level 3 Problem 13: 13_DenseNet121TransitionLayer.py

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
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(Model, self).__init__()
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

batch_size = 128
num_input_features = 32
num_output_features = 64
height, width = 256, 256

def get_inputs():
    return [torch.rand(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, num_output_features]

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

A previous single-shot attempt failed. Error: `error: HIP error: invalid argument
Search for `hipErrorInvalidValue' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__HIPRT__TYPES.html for more information.
HIP kernel errors might be asynchro`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_transition_kernel(
    x_ptr,
    out_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_running_mean_ptr,
    bn_running_var_ptr,
    conv_weight_ptr,
    stride_x_batch,
    stride_x_channel,
    stride_x_height,
    stride_x_width,
    stride_out_batch,
    stride_out_channel,
    stride_out_height,
    stride_out_width,
    num_input_channels,
    num_output_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    eps,
    BLOCK_SIZE_OUTPUT_CHANNEL: tl.constexpr,
    BLOCK_SIZE_INPUT_CHANNEL: tl.constexpr,
):
    # Each program handles one output element (batch, output_channel, out_h, out_w)
    pid = tl.program_id(0)
    
    # Decode the output position
    num_output_elements = output_height * output_width
    batch_idx = pid // (num_output_channels * num_output_elements)
    remaining = pid % (num_output_channels * num_output_elements)
    out_channel_idx = remaining // num_output_elements
    spatial_idx = remaining % num_output_elements
    out_h = spatial_idx // output_width
    out_w = spatial_idx % output_width
    
    # Calculate input coordinates for 2x2 avg pool with stride 2
    in_h_start = out_h * 2
    in_w_start = out_w * 2
    
    # Load BatchNorm parameters for this output channel
    bn_weight = tl.load(bn_weight_ptr + out_channel_idx)
    bn_bias = tl.load(bn_bias_ptr + out_channel_idx)
    bn_mean = tl.load(bn_running_mean_ptr + out_channel_idx)
    bn_var = tl.load(bn_running_var_ptr + out_channel_idx)
    
    # Compute BatchNorm scale factor
    bn_scale = bn_weight / tl.sqrt(bn_var + eps)
    
    # Accumulator for conv + pool
    acc = tl.zeros([BLOCK_SIZE_INPUT_CHANNEL], dtype=tl.float32)
    
    # Process input channels in blocks
    for ic_start in range(0, num_input_channels, BLOCK_SIZE_INPUT_CHANNEL):
        ic_offsets = ic_start + tl.arange(0, BLOCK_SIZE_INPUT_CHANNEL)
        ic_mask = ic_offsets < num_input_channels
        
        # Load conv weights for this output channel and input channels
        conv_weights = tl.load(
            conv_weight_ptr + out_channel_idx * num_input_channels + ic_offsets,
            mask=ic_mask,
            other=0.0
        )
        
        # Load 4 input positions for avg pool and apply batchnorm + relu
        pool_sum = tl.zeros([BLOCK_SIZE_INPUT_CHANNEL], dtype=tl.float32)
        
        for dh in range(2):
            for dw in range(2):
                in_h = in_h_start + dh
                in_w = in_w_start + dw
                
                # Check bounds
                in_h_valid = in_h < input_height
                in_w_valid = in_w < input_width
                
                if in_h_valid and in_w_valid:
                    # Load input values for all input channels
                    x_offsets = (
                        batch_idx * stride_x_batch +
                        ic_offsets * 
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 13
```
