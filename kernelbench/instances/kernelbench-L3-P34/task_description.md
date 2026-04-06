# KernelBench Level 3 Problem 34: 34_VanillaRNNHidden.py

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
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.

        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN.

        :param x: Input tensor of shape (seq_len, batch_size, input_size)
        :param h0: Initial hidden state tensor of shape (batch_size, hidden_size)
        :return: Output tensor of shape (seq_len, batch_size, output_size)
        """
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []

        for t in range(seq_len):
            combined = torch.cat((x[t], hidden), dim=1)  # Concatenate input and hidden state
            hidden = self.tanh(self.i2h(combined))  # Update hidden state
            output = self.h2o(hidden)  # Compute output
            outputs.append(output)

        return torch.stack(outputs, dim=0)  # (seq_len, batch_size, output_size)

# === Test configuration ===
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [
        torch.rand(sequence_length, batch_size, input_size),
        torch.rand(batch_size, hidden_size)
    ]

def get_init_inputs():
    return [input_size, hidden_size, output_size]

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

A previous single-shot attempt failed. Error: `at 27:4:
    col_mask = col_offsets < hidden_size

    input_offset = row_idx * input_size
    hidden_offset = row_idx * hidden_size
    out_offset = row_idx * hidden_size

    input_offsets = tl.a...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_rnn_cell_kernel(
    x_ptr,
    h_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE_OUT
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_OUT)
    col_mask = col_offsets < hidden_size

    input_offset = row_idx * input_size
    hidden_offset = row_idx * hidden_size
    out_offset = row_idx * hidden_size

    input_offsets = tl.arange(0, BLOCK_SIZE_IN)
    hidden_offsets = tl.arange(0, BLOCK_SIZE_IN)

    acc = tl.zeros([BLOCK_SIZE_OUT], dtype=tl.float32)

    for i_start in range(0, input_size, BLOCK_SIZE_IN):
        i_offsets = i_start + input_offsets
        i_mask = i_offsets < input_size
        x_vals = tl.load(x_ptr + input_offset + i_offsets, mask=i_mask, other=0.0)
        w_vals = tl.load(
            w_ptr + i_offsets * (input_size + hidden_size) + col_offsets,
            mask=i_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x_vals[None, :], w_vals).to(tl.float32)

    for h_start in range(0, hidden_size, BLOCK_SIZE_IN):
        h_offsets = h_start + input_offsets
        h_mask = h_offsets < hidden_size
        h_vals = tl.load(h_ptr + hidden_offset + h_offsets, mask=h_mask, other=0.0)
        w_vals = tl.load(
            w_ptr + (input_size + h_offsets) * (input_size + hidden_size) + col_offsets,
            mask=h_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(h_vals[None, :], w_vals).to(tl.float32)

    bias_vals = tl.load(b_ptr + col_offsets, mask=col_mask, other=0.0)
    acc += bias_vals
    out_vals = tl.math.tanh(acc)
    tl.store(out_ptr + out_offset + col_offsets, out_vals, mask=col_mask)


@triton.jit
def fused_linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE_OUT
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_OUT)
    col_mask = col_offsets < out_features

    row_offset = row_idx * in_features
    out_offset = row_idx * out_features

    in_offsets = tl.arange(0, BLOCK_SIZE_IN)
    acc = tl.zeros([BLOCK_SIZE_OUT], dtype=tl.float32)

    for i_start in range(0, in_features, BLOCK_SIZE_IN):
        i_offsets = i_start + in_offsets
        i_mask = i_offsets < in_features
        x_vals = tl.load(x_ptr + row_offset + i_offsets, mask=i_mask, other=0.0)
        w_vals = tl.load(
            w_ptr + i_offsets * out_features + col_offsets,
            mask=i_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
     
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 34
```
