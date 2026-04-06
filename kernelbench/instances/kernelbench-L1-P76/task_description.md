# KernelBench Level 1 Problem 76: 76_conv_standard_1D_dilated_strided__.py

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
    Performs a standard 1D convolution operation with asymmetric input and a square kernel, potentially dilated and strided.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        return self.conv1d(x)

# Test code
batch_size = 64
in_channels = 64
out_channels = 128
kernel_size = 3
# longer signal
length = 524280
stride = 3
dilation = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, dilation]
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
def conv1d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_length: tl.constexpr,
    output_length: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Each program handles one (batch, out_channel, out_position)
    pid = tl.program_id(0)
    
    # Decode program ID
    out_pos = pid % output_length
    pid_temp = pid // output_length
    out_ch = pid_temp % out_channels
    batch_idx = pid_temp // out_channels
    
    # Calculate input start position
    input_start = out_pos * stride
    
    # Accumulator
    acc = 0.0
    
    # Loop over input channels in blocks
    for ic_start in range(0, in_channels, BLOCK_IC):
        ic_offsets = ic_start + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offsets < in_channels
        
        # Loop over kernel positions
        for k in range(kernel_size):
            input_pos = input_start + k * dilation
            pos_mask = input_pos < input_length
            
            # Combined mask
            mask = ic_mask & pos_mask
            
            # Load input: [BLOCK_IC]
            x_offsets = (batch_idx * in_channels * input_length + 
                        ic_offsets * input_length + 
                        input_pos)
            x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            
            # Load weights: [BLOCK_IC]
            w_offsets = out_ch * in_channels * kernel_size + ic_offsets * kernel_size + k
            w = tl.load(weight_ptr + w_offsets, mask=ic_mask, other=0.0)
            
            # Multiply and accumulate
            acc += tl.sum(x * w, axis=0)
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + out_ch)
        acc += bias
    
    # Store output
    out_offset = batch_idx * out_channels * output_length + out_ch * output_length + out_pos
    tl.store(out_ptr + out_offset, acc)


def triton_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    dilation: int,
) -> torch.Tensor:
    """
    Triton-based 1D convolution.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = weight.shape
    
    # Calculate output length
    output_length = (input_length - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, output_length, dtype=x.dtype, device=x.device)
    
    # Prepare bias
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous()
    else:
        bias = torch
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 76
```
