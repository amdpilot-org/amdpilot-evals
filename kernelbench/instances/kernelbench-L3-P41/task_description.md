# KernelBench Level 3 Problem 41: 41_GRUBidirectional.py

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
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
    
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

A previous single-shot attempt failed. Error: `at 51:8:
        other=0.0
    ).to(tl.float32)

    n_input = tl.load(
        gate_input_ptr + pid * (3 * hidden_size) + 2 * hidden_size + offs,
        mask=mask,
        other=0.0
    ).to(tl.f...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gru_gate_fusion_kernel(
    gate_input_ptr,
    h_prev_ptr,
    h_new_ptr,
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GRU gate computation and hidden state update.
    gate_input contains [r_gate, z_gate, n_gate] concatenated.
    Computes:
        r = sigmoid(r_gate)
        z = sigmoid(z_gate)
        n = tanh(n_gate)
        h_new = (1 - z) * n + z * h_prev
    """
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    h_prev = tl.load(
        h_prev_ptr + pid * hidden_size + offs,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    r_input = tl.load(
        gate_input_ptr + pid * (3 * hidden_size) + offs,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    z_input = tl.load(
        gate_input_ptr + pid * (3 * hidden_size) + hidden_size + offs,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    n_input = tl.load(
        gate_input_ptr + pid * (3 * hidden_size) + 2 * hidden_size + offs,
        mask=mask,
        other=0.0
    ).to(tl.float32)

    r = tl.sigmoid(r_input)
    z = tl.sigmoid(z_input)
    n = tl.libdevice.tanh(n_input)

    h_new = (1.0 - z) * n + z * h_prev

    tl.store(
        h_new_ptr + pid * hidden_size + offs,
        h_new,
        mask=mask
    )


def gru_gate_fusion(gate_input, h_prev):
    """
    Wrapper for GRU gate fusion kernel.
    gate_input: (batch, 3*hidden)
    h_prev: (batch, hidden)
    Returns: (batch, hidden)
    """
    batch_size = gate_input.shape[0]
    hidden_size = gate_input.shape[1] // 3

    gate_input = gate_input.contiguous()
    h_prev = h_prev.contiguous()

    h_new = torch.empty(
        (batch_size, hidden_size),
        dtype=gate_input.dtype,
        device=gate_input.device
    )

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    grid = (batch_size,)

    gru_gate_fusion_kernel[grid](
        gate_input,
        h_prev,
        h_new,
        batch_size=batch_size,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return h_new


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2

        self._init_gru_weights()

    def _init_gru_weights(self):
        num_directions = self.num_directions

        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = f'_l{layer}' if layer > 0 else ''
                if direction == 1:
                    suffix += '_reverse'

       
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 41
```
