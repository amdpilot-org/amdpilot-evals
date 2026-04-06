# KernelBench Level 3 Problem 4: 4_LeNet5.py

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
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model (larger batch & image)
batch_size = 4096
num_classes = 20

def get_inputs():
    return [torch.rand(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]
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

A previous single-shot attempt failed. Error: `at 35:12:
    stride_oh,
    stride_ow,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
)...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def linear_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xn,
    stride_wk,
    stride_wn,
    stride_om,
    stride_ok,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    m_range = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_range = n_start + tl.arange(0, BLOCK_SIZE_N)

    m_mask = m_range < M
    n_mask = n_range < K

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_SIZE_K):
        k_range = k + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_range < N

        x_ptrs = x_ptr + m_range[:, None] * stride_xm + k_range[None, :] * stride_xn
        x = tl.load(x_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)

        w_ptrs = weight_ptr + n_range[None, :] * stride_wk + k_range[:, None] * stride_wn
        w = tl.load(w_ptrs, mask=(n_mask[None, :] & k_mask[:, None]), other=0.0)

        acc += tl.dot(x, w)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_range, mask=n_mask, other=0.0)
        acc += bias[None, :]

    acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + m_range[:, None] * stride_om + n_range[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


def triton_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    M, N = x.shape
    K = weight.shape[0]

    out = torch.empty((M, K), dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(K, meta['BLOCK_SIZE_N']),
    )

    linear_relu_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


@triton.jit
def conv2d_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    K,
    kernel_size,
    stride,
    padding,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    stride_k,
    stride_cw,
    stride_kh,
    stride_kw,
    stride_on,
    stride_ok,
    stride_oh,
    stride_ow,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 4
```
