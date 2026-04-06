# KernelBench Level 1 Problem 50: 50_conv_standard_2D__square_input__square_kernel.py

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
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

# Test code
batch_size = 256
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, 3, 224, 224)]

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

A previous single-shot attempt failed. Error: `at 59:4:
                    )
                    weight_val = tl.load(weight_ptr + weight_offset)

                    acc += input_val * weight_val

    output_offset = (
        n_idx * stride_...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N, C, H, W,
    K, kH, kW,
    stride, padding,
    H_out, W_out,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_k, stride_w_c, stride_w_h, stride_w_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_nk = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    n_idx = pid_nk // K
    k_idx = pid_nk % K
    
    h_out_idx = pid_hw // W_out
    w_out_idx = pid_hw % W_out
    
    acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
    
    for c in range(C):
        for kh in range(kH):
            for kw in range(kW):
                h_in = h_out_idx * stride + kh - padding
                w_in = w_out_idx * stride + kw - padding
                
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                
                if in_bounds:
                    input_offset = (
                        n_idx * stride_in_n +
                        c * stride_in_c +
                        h_in * stride_in_h +
                        w_in * stride_in_w
                    )
                    input_val = tl.load(input_ptr + input_offset)
                    
                    weight_offset = (
                        k_idx * stride_w_k +
                        c * stride_w_c +
                        kh * stride_w_h +
                        kw * stride_w_w
                    )
                    weight_val = tl.load(weight_ptr + weight_offset)
                    
                    acc += input_val * weight_val
    
    output_offset = (
        n_idx * stride_out_n +
        k_idx * stride_out_c +
        h_out_idx * stride_out_h +
        w_out_idx * stride_out_w
    )
    tl.store(output_ptr + output_offset, acc)


def triton_conv2d(input_tensor, weight_tensor, bias_tensor=None, stride=4, padding=2):
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Tensors must be on CUDA."
    
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    
    N, C, H, W = input_tensor.shape
    K, _, kH, kW = weight_tensor.shape
    
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    
    output_tensor = torch.empty((N, K, H_out, W_out), dtype=input_tensor.dtype, device=input_tensor.device)
    
    stride_in_n = input_tensor.stride(0)
    stride_in_c = input_tensor.stride(1)
    stride_in_h = input_tensor.stride(2)
    stride_in_w = input_tensor.stride(3)
    
    stride_w_k = weight_tensor.stride(0)
    stride_w_c = weight_tensor.stride(1)
    stride_w_h = weight_tensor.stride(2)
    stride_w_w = weight_tensor.stride(3)
    
    stride_out_n = output_tensor.stride(0)
    stride_out_c = output_tensor.stride(1)
    stride_out_h = output_ten
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 50
```
