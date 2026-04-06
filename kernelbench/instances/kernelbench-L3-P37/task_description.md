# KernelBench Level 3 Problem 37: 37_LSTMCn.py

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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(Model, self).__init__()
        # Initialize hidden state with random values
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """
        
        # Forward propagate LSTM
        out, state = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out: tensor of shape (batch_size, output_size)
        
        return state[1]

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.rand(batch_size, sequence_length, input_size),torch.rand((num_layers, batch_size, hidden_size)),torch.rand((num_layers, batch_size, hidden_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
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

A previous single-shot attempt failed. Error: `at 63:13:
        f_gate += tl.sum(h_val * w_f)
        g_gate += tl.sum(h_val * w_g)
        o_gate += tl.sum(h_val * w_o)

    i_gate += tl.load(b_ptr + hidden_idx)
    f_gate += tl.load(b_ptr + ...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def lstm_cell_kernel(
    x_ptr, h_prev_ptr, c_prev_ptr,
    w_ih_ptr, w_hh_ptr, b_ptr,
    h_out_ptr, c_out_ptr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    input_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // hidden_size
    hidden_idx = pid % hidden_size
    
    if batch_idx >= batch_size or hidden_idx >= hidden_size:
        return
    
    h_prev = tl.load(h_prev_ptr + batch_idx * hidden_size + hidden_idx)
    c_prev = tl.load(c_prev_ptr + batch_idx * hidden_size + hidden_idx)
    
    i_gate = 0.0
    f_gate = 0.0
    g_gate = 0.0
    o_gate = 0.0
    
    for k in range(0, input_size, BLOCK_I):
        offsets = k + tl.arange(0, BLOCK_I)
        mask = offsets < input_size
        x_val = tl.load(x_ptr + batch_idx * input_size + offsets, mask=mask, other=0.0)
        
        w_i = tl.load(w_ih_ptr + hidden_idx * input_size + offsets, mask=mask, other=0.0)
        w_f = tl.load(w_ih_ptr + (hidden_size + hidden_idx) * input_size + offsets, mask=mask, other=0.0)
        w_g = tl.load(w_ih_ptr + (2 * hidden_size + hidden_idx) * input_size + offsets, mask=mask, other=0.0)
        w_o = tl.load(w_ih_ptr + (3 * hidden_size + hidden_idx) * input_size + offsets, mask=mask, other=0.0)
        
        i_gate += tl.sum(x_val * w_i)
        f_gate += tl.sum(x_val * w_f)
        g_gate += tl.sum(x_val * w_g)
        o_gate += tl.sum(x_val * w_o)
    
    for k in range(0, hidden_size, BLOCK_H):
        offsets = k + tl.arange(0, BLOCK_H)
        mask = offsets < hidden_size
        h_val = tl.load(h_prev_ptr + batch_idx * hidden_size + offsets, mask=mask, other=0.0)
        
        w_i = tl.load(w_hh_ptr + hidden_idx * hidden_size + offsets, mask=mask, other=0.0)
        w_f = tl.load(w_hh_ptr + (hidden_size + hidden_idx) * hidden_size + offsets, mask=mask, other=0.0)
        w_g = tl.load(w_hh_ptr + (2 * hidden_size + hidden_idx) * hidden_size + offsets, mask=mask, other=0.0)
        w_o = tl.load(w_hh_ptr + (3 * hidden_size + hidden_idx) * hidden_size + offsets, mask=mask, other=0.0)
        
        i_gate += tl.sum(h_val * w_i)
        f_gate += tl.sum(h_val * w_f)
        g_gate += tl.sum(h_val * w_g)
        o_gate += tl.sum(h_val * w_o)
    
    i_gate += tl.load(b_ptr + hidden_idx)
    f_gate += tl.load(b_ptr + hidden_size + hidden_idx)
    g_gate += tl.load(b_ptr + 2 * hidden_size + hidden_idx)
    o_gate += tl.load(b_ptr + 3 * hidden_size + hidden_idx)
    
    i_gate = tl.sigmoid(i_gate)
    f_gate = tl.sigmoid(f_gate)
    g_gate = tl.tanh(g_gate)
    o_gate = tl.sigmoid(o_gate)
    
    c_out = f_gate * c_prev + i_gate * g_gate
    h_out = o_gate * tl.tanh(c_out)
    
    tl.store(h_out_ptr + batch_idx * hidden_size + hidden_idx, h_out)
    tl.store(c_out_ptr + batch_idx * hidden_size + hidden_idx, c_out)


@triton.j
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 37
```
