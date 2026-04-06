# KernelBench Level 1 Problem 41: 41_Max_Pooling_1D.py

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
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(Model, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied, shape (batch_size, num_features, output_sequence_length).
        """
        return self.maxpool(x)

batch_size = 64
features = 192
sequence_length = 65536

kernel_size = 8
stride      = 1
padding     = 4
dilation    = 3            

return_indices = False

def get_inputs():
    x = torch.rand(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]
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

A previous single-shot attempt failed. Error: `at 26:12:
):
    # Each program handles BLOCK_SIZE output positions
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Process each output position in this block
    for i in rang...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,           # Pointer to input tensor
    out_ptr,         # Pointer to output tensor
    indices_ptr,     # Pointer to indices tensor (optional)
    batch_size: tl.constexpr,
    num_features: tl.constexpr,
    input_length: tl.constexpr,
    output_length: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    return_indices: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE output positions
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process each output position in this block
    for i in range(BLOCK_SIZE):
        output_pos = block_start + i
        
        # Check if this output position is valid
        if output_pos >= output_length:
            break
        
        # Calculate batch and feature indices
        # Flatten: (batch, feature, output_pos) -> linear index
        batch_idx = output_pos // (num_features * output_length)
        remaining = output_pos % (num_features * output_length)
        feature_idx = remaining // output_length
        out_seq_pos = remaining % output_length
        
        # Calculate the input position range for this output position
        input_start = out_seq_pos * stride - padding
        
        # Initialize max value and index
        max_val = -1e30  # Very small number for FP32
        max_idx = 0
        
        # Iterate over the kernel window
        for k in range(kernel_size):
            input_pos = input_start + k * dilation
            
            # Check if input position is within bounds
            if input_pos >= 0 and input_pos < input_length:
                # Calculate the flat index for loading
                input_idx = batch_idx * num_features * input_length + feature_idx * input_length + input_pos
                
                # Load the value
                val = tl.load(x_ptr + input_idx)
                
                # Update max if this value is larger
                if val > max_val:
                    max_val = val
                    max_idx = input_pos
        
        # Calculate output index
        output_idx = batch_idx * num_features * output_length + feature_idx * output_length + out_seq_pos
        
        # Store the result
        tl.store(out_ptr + output_idx, max_val)
        
        # Store indices if requested
        if return_indices:
            tl.store(indices_ptr + output_idx, max_idx)


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    return_indices: bool = False
):
    """
    Triton-based 1D Max Pooling implementation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    batch_size, num_features, input_length = x.sha
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 41
```
