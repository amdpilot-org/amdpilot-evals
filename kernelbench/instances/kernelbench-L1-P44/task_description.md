# KernelBench Level 1 Problem 44: 44_Average_Pooling_1D.py

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
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool(x)

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]
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

A previous single-shot attempt failed. Error: `at 31:12:
    # Starting output position for this block
    output_start = tl.program_id(1) * BLOCK_SIZE

    # Create offsets for this block
    output_offsets = output_start + tl.arange(0, BLOCK_...`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    input_length: tl.constexpr,
    output_length: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE output elements along the length dimension
    pid = tl.program_id(0)
    
    # Decode program ID into batch and channel
    channel = pid % in_channels
    batch = pid // in_channels
    
    # Starting output position for this block
    output_start = tl.program_id(1) * BLOCK_SIZE
    
    # Create offsets for this block
    output_offsets = output_start + tl.arange(0, BLOCK_SIZE)
    mask = output_offsets < output_length
    
    # For each output position in the block, compute average pooling
    for i in range(BLOCK_SIZE):
        output_pos = output_start + i
        if output_pos >= output_length:
            break
        
        # Calculate the starting position in the input for this output position
        input_start = output_pos * stride - padding
        
        # Accumulate values in the kernel window
        acc = 0.0
        count = 0
        
        for k in range(kernel_size):
            input_idx = input_start + k
            # Check bounds
            if input_idx >= 0 and input_idx < input_length:
                # Calculate input offset
                input_offset = batch * in_channels * input_length + channel * input_length + input_idx
                val = tl.load(x_ptr + input_offset)
                acc += val
                count += 1
        
        # Compute average
        if count > 0:
            avg = acc / count
        else:
            avg = 0.0
        
        # Store result
        out_offset = batch * in_channels * output_length + channel * output_length + output_pos
        tl.store(out_ptr + out_offset, avg)


def triton_avg_pool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    """
    Triton-based 1D Average Pooling implementation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output length
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Prepare output tensor
    out = torch.empty((batch_size, in_channels, output_length), dtype=x.dtype, device=x.device)
    
    # Block size for output length dimension
    BLOCK_SIZE = 64
    
    # Grid: one program per (batch, channel) pair, multiple blocks along output length
    grid = (
        batch_size * in_channels,
        (output_length + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    
    # Launch kernel
    avg_pool1d_kernel[grid](
        x,
        out,
        batch_size,
        in_channels,
        input_length,
        output_lengt
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 44
```
