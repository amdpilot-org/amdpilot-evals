# KernelBench Level 1 Problem 64: 64_conv_transposed_1D.py

## Goal

Write an optimized Triton kernel implementation (`ModelNew`) that:
1. Produces the **exact same output** as the PyTorch reference `Model`
2. Is **faster** than the PyTorch baseline
3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)

## PyTorch Reference Implementation

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d_transpose(x)

# Test code
batch_size = 64
in_channels = 128
out_channels = 128
kernel_size = 3
# much larger signal length for heavier workload
length = 65536

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization
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

A previous single-shot attempt failed. Error: `at 59:12:
        ic_base = (oc_offsets // groups) * groups
        ic_mask = ic_base < in_channels

        w_off = (
            oc_offsets * weight_out_stride +
            ic_base * weight_in_s...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    groups,
    input_batch_stride,
    input_channel_stride,
    input_length_stride,
    weight_out_stride,
    weight_in_stride,
    weight_k_stride,
    output_batch_stride,
    output_channel_stride,
    output_length_stride,
    HAS_BIAS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_LEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ol = tl.program_id(2)

    oc_start = pid_oc * BLOCK_OUT
    oc_offsets = oc_start + tl.arange(0, BLOCK_OUT)
    oc_mask = oc_offsets < out_channels

    ol_start = pid_ol * BLOCK_LEN
    ol_offsets = ol_start + tl.arange(0, BLOCK_LEN)
    ol_mask = ol_offsets < output_length

    acc = tl.zeros([BLOCK_OUT, BLOCK_LEN], dtype=tl.float32)

    for k in range(kernel_size):
        il_raw = (ol_offsets + padding - k) 
        il_valid = (il_raw >= 0) & (il_raw < input_length * stride) & ((il_raw % stride) == 0)
        il = (il_raw // stride).to(tl.int32)
        il_mask = il_valid

        ic_base = (oc_offsets // groups) * groups
        ic_mask = ic_base < in_channels

        w_off = (
            oc_offsets * weight_out_stride +
            ic_base * weight_in_stride +
            k * weight_k_stride
        )
        w = tl.load(weight_ptr + w_off, mask=oc_mask & ic_mask, other=0.0)

        i_off = (
            pid_b * input_batch_stride +
            ic_base * input_channel_stride +
            il * input_length_stride
        )
        i_val = tl.load(input_ptr + i_off, mask=oc_mask & ic_mask & il_mask[None, :], other=0.0)

        acc += w[:, None] * i_val

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0)
        acc += bias[:, None]

    o_off = (
        pid_b * output_batch_stride +
        oc_offsets[:, None] * output_channel_stride +
        ol_offsets[None, :] * output_length_stride
    )
    tl.store(output_ptr + o_off, acc, mask=oc_mask[:, None] & ol_mask[None, :])


def triton_conv_transpose1d(x, weight, bias, stride, padding, output_padding, groups):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, input_length = x.shape
    out_channels, in_channels_per_group, kernel_size = weight.shape
    output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding

    out = torch.empty((batch_size, out_channels, output_length), dtype=x.dtype, device=x.device)

    input_batch_stride = x.stride(0)
    input_channel_stride = x.stride(1)
    input_length_stride = x.stride(2)

    weight_out_stride = wei
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 64
```
