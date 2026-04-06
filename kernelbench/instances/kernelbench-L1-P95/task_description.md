# KernelBench Level 1 Problem 95: 95_CrossEntropyLoss.py

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
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []

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

A previous single-shot attempt failed. Error: `error: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    predictions_ptr,
    targets_ptr,
    loss_ptr,
    stride_pred_batch,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    
    target = tl.load(targets_ptr + batch_idx)
    
    max_val = -1e10
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        logits = tl.load(
            predictions_ptr + batch_idx * stride_pred_batch + offsets,
            mask=mask,
            other=-1e10
        )
        block_max = tl.max(logits, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    sum_exp = 0.0
    target_logit = 0.0
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        logits = tl.load(
            predictions_ptr + batch_idx * stride_pred_batch + offsets,
            mask=mask,
            other=-1e10
        )
        
        exp_vals = tl.exp(logits - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)
        
        target_mask = offsets == target
        target_logit += tl.sum(tl.where(target_mask, logits, 0.0), axis=0)
    
    loss = -target_logit + tl.log(sum_exp) + max_val
    tl.store(loss_ptr + batch_idx, loss)


def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor):
    assert predictions.is_cuda and targets.is_cuda, "Tensors must be on CUDA."
    assert predictions.dim() == 2, "Predictions must be 2D (batch_size, num_classes)."
    assert targets.dim() == 1, "Targets must be 1D (batch_size,)."
    assert predictions.shape[0] == targets.shape[0], "Batch size mismatch."
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    num_classes = predictions.shape[1]
    
    loss = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    BLOCK_SIZE = 256
    grid = (batch_size,)
    
    cross_entropy_kernel[grid](
        predictions,
        targets,
        loss,
        predictions.stride(0),
        num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return loss.mean()


class ModelNew(nn.Module):
    """
    Optimized model that computes Cross Entropy Loss using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 1 --problem-id 95
```
