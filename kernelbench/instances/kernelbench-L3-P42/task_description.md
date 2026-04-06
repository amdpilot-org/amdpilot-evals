# KernelBench Level 3 Problem 42: 42_GRUBidirectionalHidden.py

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
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(Model, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
    
    def forward(self, x,h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size) (default: None)
        :return: output, h_n
            - output: The output features (h_t) from the last layer of the GRU, for each t, shape (seq_len, batch_size, num_directions * hidden_size) if batch_first=False, otherwise (batch_size, seq_len, num_directions * hidden_size)
            - h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        output, h_n = self.gru(x, h0)
        return h_n

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.rand(seq_len, batch_size, input_size),torch.rand((num_layers*2, batch_size, hidden_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
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

A previous single-shot attempt failed. Error: `at 71:13:
    w_ih_r = tl.load(w_ih_ptr + hidden_offsets * input_size + input_offsets, 
                     mask=(hidden_mask[:, None] & input_mask[None, :]), other=0.0)

    # For simplicity, use...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gru_cell_kernel(
    # Input pointers
    x_ptr, h_prev_ptr,
    # Weight pointers
    w_ih_ptr, w_hh_ptr,
    # Bias pointers
    b_ih_ptr, b_hh_ptr,
    # Output pointer
    h_new_ptr,
    # Dimensions
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    # Strides
    stride_x_batch: tl.constexpr,
    stride_h_batch: tl.constexpr,
    stride_h_new_batch: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Triton kernel for GRU cell computation.
    Computes reset, update, and new gates, then the new hidden state.
    Each program handles one batch element.
    """
    pid = tl.program_id(0)
    
    # Create offsets for hidden dimension
    hidden_offsets = tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    
    # Create offsets for input dimension
    input_offsets = tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < input_size
    
    # Load input x for this batch
    x_ptrs = x_ptr + pid * stride_x_batch + input_offsets
    x = tl.load(x_ptrs, mask=input_mask, other=0.0)
    
    # Load previous hidden state
    h_prev_ptrs = h_prev_ptr + pid * stride_h_batch + hidden_offsets
    h_prev = tl.load(h_prev_ptrs, mask=hidden_mask, other=0.0)
    
    # Initialize gate accumulators
    r_gate = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    z_gate = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    n_gate = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute gates using simplified approach (for demonstration)
    # In production, you'd do proper matrix multiplication with blocks
    # Here we use element-wise operations for clarity
    
    # Reset gate: r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    # Update gate: z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    # New gate: n = tanh(W_in @ x + b_in + W_hn @ (r * h) + b_hn)
    
    # Simplified computation (assuming weights are pre-processed)
    # Load weight contributions (simplified for this example)
    w_ih_r = tl.load(w_ih_ptr + hidden_offsets * input_size + input_offsets, 
                     mask=(hidden_mask[:, None] & input_mask[None, :]), other=0.0)
    
    # For simplicity, use element-wise multiplication as approximation
    # In real implementation, you'd need proper matrix multiplication
    x_contrib = tl.sum(x * x, axis=0)  # Simplified
    h_contrib = tl.sum(h_prev * h_prev, axis=0)  # Simplified
    
    # Gate computations with sigmoid/tanh
    r_gate = 1.0 / (1.0 + tl.exp(-(x_contrib + h_contrib)))
    z_gate = 1.0 / (1.0 + tl.exp(-(x_contrib + h_contrib)))
    n_gate = tl.libdevice.tanh(x_contrib + h_contrib)
    
    # Compute new hidden state: h_new = (1 - z) * n + z * h_prev
    h_new = (1.0 - z_gate) * n_gate + z_gate * h_prev
    
    # Store output
    h_new_ptrs = h_new_ptr + pid * stride_h_new_batch + hidd
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 42
```
