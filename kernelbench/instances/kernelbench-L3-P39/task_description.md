# KernelBench Level 3 Problem 39: 39_GRU.py

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
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
    
    def forward(self, x,h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size) (default: None)
        :return: output, h_n
            - output: The output features (h_t) from the last layer of the GRU, for each t, shape (seq_len, batch_size, num_directions * hidden_size) if batch_first=False, otherwise (batch_size, seq_len, num_directions * hidden_size)
            - h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        output, h_n = self.gru(x, h0)
        return output

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.rand(seq_len, batch_size, input_size),torch.rand((num_layers, batch_size, hidden_size))]

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

A previous single-shot attempt failed. Error: `at 36:19:

        acc = tl.zeros([BLOCK_HIDDEN], dtype=tl.float32)

        # Input-hidden matmul
        for i_block in range(0, input_size, BLOCK_INPUT):
            i_offset = i_block + tl.aran...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gru_activation_kernel(
    gates_ptr,      # (batch, 3*hidden) - [reset, update, new]
    h_prev_ptr,     # (batch, hidden)
    h_new_ptr,      # (batch, hidden)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    hidden_offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offset < hidden_size

    reset_gate = tl.load(gates_ptr + batch_idx * 3 * hidden_size + hidden_offset,
                         mask=hidden_mask, other=0.0)
    update_gate = tl.load(gates_ptr + batch_idx * 3 * hidden_size + hidden_size + hidden_offset,
                          mask=hidden_mask, other=0.0)
    new_gate = tl.load(gates_ptr + batch_idx * 3 * hidden_size + 2 * hidden_size + hidden_offset,
                       mask=hidden_mask, other=0.0)

    reset_gate = tl.sigmoid(reset_gate)
    update_gate = tl.sigmoid(update_gate)
    new_gate = tl.tanh(new_gate)

    h_prev = tl.load(h_prev_ptr + batch_idx * hidden_size + hidden_offset,
                     mask=hidden_mask, other=0.0)

    h_new = (1 - update_gate) * new_gate + update_gate * h_prev

    tl.store(h_new_ptr + batch_idx * hidden_size + hidden_offset, h_new, mask=hidden_mask)


def fused_gru_activation(gates: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    batch_size, total_hidden = gates.shape
    hidden_size = total_hidden // 3

    assert gates.is_cuda and h_prev.is_cuda, "Tensors must be on CUDA"
    gates = gates.contiguous()
    h_prev = h_prev.contiguous()

    h_new = torch.empty((batch_size, hidden_size), dtype=gates.dtype, device=gates.device)

    BLOCK_SIZE = 64
    grid = (batch_size, (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE)

    fused_gru_activation_kernel[grid](
        gates, h_prev, h_new,
        batch_size, hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return h_new


@triton.jit
def fused_gru_gates_kernel(
    x_ptr,          # (batch, input_size)
    h_prev_ptr,     # (batch, hidden_size)
    w_ih_ptr,       # (3*hidden, input_size)
    w_hh_ptr,       # (3*hidden, hidden_size)
    b_ih_ptr,       # (3*hidden,)
    b_hh_ptr,       # (3*hidden,)
    gates_ptr,      # (batch, 3*hidden)
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_INPUT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    hidden_idx = tl.program_id(1)

    hidden_offset = hidden_idx * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden_offset < hidden_size

    for gate in range(3):
        gate_offset = gate * hidden_size + hidden_offset
        gate_mask = hidden_offset < hidden_size

        acc = tl.zeros([BLOCK_HIDDEN], dtype=tl.float32)

        # Input-hidden matmul
        for i_block in range(0, input_size, BLOCK_IN
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 39
```
