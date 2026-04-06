# KernelBench Level 2 Problem 18: 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py

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
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = self.linear(x)  # (batch_size, out_features)
        x = torch.sum(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.max(x, dim=1, keepdim=True)[0] # (batch_size, 1)
        x = torch.mean(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        x = torch.logsumexp(x, dim=1, keepdim=True) # (batch_size, 1)
        return x

batch_size = 1024
in_features  = 8192  
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
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

A previous single-shot attempt failed. Error: `at 44:4:
        # Load pre-computed weight sums
        w_sum = tl.load(weight_sum_ptr + offsets, mask=mask, other=0.0)

        # Accumulate dot product
        acc += tl.sum(x * w_sum)

    # Ad...`

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_sum_kernel(
    x_ptr,
    weight_sum_ptr,
    bias_sum_ptr,
    out_ptr,
    batch_size,
    in_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: out[i] = sum_j (linear(x)[i, j])
    Which simplifies to: out[i] = x[i] @ weight_sum + bias_sum
    
    This fuses the linear layer with the sum operation, avoiding materialization
    of the full (batch_size, out_features) intermediate tensor.
    
    Note: All subsequent operations (max, mean, logsumexp) on a (batch_size, 1)
    tensor along dim=1 are identity operations and are skipped.
    """
    batch_idx = tl.program_id(0)
    
    # Accumulator for the dot product
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Iterate over input features in blocks
    for start in range(0, in_features, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < in_features
        
        # Load input features for this batch element
        x = tl.load(x_ptr + batch_idx * in_features + offsets, mask=mask, other=0.0)
        
        # Load pre-computed weight sums
        w_sum = tl.load(weight_sum_ptr + offsets, mask=mask, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(x * w_sum)
    
    # Add bias sum
    bias_sum = tl.load(bias_sum_ptr)
    acc += bias_sum
    
    # Store result (shape: batch_size, 1)
    tl.store(out_ptr + batch_idx, acc)


class ModelNew(nn.Module):
    """
    Optimized model that fuses the linear layer with the sum operation.
    All subsequent operations (max, mean, logsumexp) on (batch_size, 1) 
    tensors are identity operations and are eliminated.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute weight and bias sums (maintains gradient flow for training)
        # weight_sum[k] = sum_j W[j, k] where W is (out_features, in_features)
        weight_sum = self.linear.weight.sum(dim=0).contiguous()
        bias_sum = self.linear.bias.sum()
        
        # Allocate output tensor
        out = torch.empty(batch_size, 1, dtype=torch.float32, device=x.device)
        
        # Configure kernel launch parameters
        BLOCK_SIZE = 256
        grid = (batch_size,)
        
        # Launch fused kernel
        fused_linear_sum_kernel[grid](
            x.contiguous(),
            weight_sum,
            bias_sum,
            out,
            batch_size,
            self.in_features,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 2 --problem-id 18
```
