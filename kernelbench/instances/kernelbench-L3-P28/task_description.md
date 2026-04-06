# KernelBench Level 3 Problem 28: 28_VisionTransformer.py

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
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model.

        :param image_size: The size of the input image (assumed to be square).
        :param patch_size: The size of each patch (assumed to be square).
        :param num_classes: The number of output classes.
        :param dim: The dimensionality of the embedding space.
        :param depth: The number of transformer layers.
        :param heads: The number of attention heads.
        :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
        :param channels: The number of channels in the input image (default is 3 for RGB).
        :param dropout: Dropout rate applied in the MLP.
        :param emb_dropout: Dropout rate applied to the embedded patches.
        """
        super(Model, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, img):
        """
        Forward pass of the Vision Transformer.

        :param img: The input image tensor, shape (batch_size, channels, image_size, image_size).
        :return: The output tensor, shape (batch_size, num_classes).
        """
        p = self.patch_size
        
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        x = self.patch_to_embedding(x)
        
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.rand(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]
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

A previous single-shot attempt failed. Error: `at 24:17:
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    patch_row = pid_n // (W // P)
    patch_col = pid_n % (W // P)

    img_start_h = patch_row...`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    stride_x,
    stride_y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride_x
    y_ptr += row_idx * stride_y
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(x_ptr + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    
    x_mean = x - mean
    var = tl.sum(x_mean * x_mean, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    x_norm = x_mean * rstd
    
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    y = x_norm * weight + bias
    
    tl.store(y_ptr + cols, y, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layer_norm_kernel[grid](
        x, out, weight, bias,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def fused_mlp_kernel(
    x_ptr,
    out_ptr,
    weight1_ptr,
    bias1_ptr,
    weight2_ptr,
    bias2_ptr,
    stride_x,
    stride_out,
    M,
    N,  # hidden dim
    K,  # output dim
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    m_offs = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
    k_offs = tl.arange(0, BLOCK_SIZE_K)
    
    mask_m = m_offs < M
    mask_n = n_offs < N
    mask_k = k_offs < K
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs_curr = k + k_offs
        mask_k_curr = k_offs_curr < K
        
        x = tl.load(x_ptr + m_offs[:, None] * K + k_offs_curr[None, :],
                    mask=mask_m[:, None] & mask_k_curr[None, :], other=0.0)
        w1 = tl.load(weight1_ptr + k_offs_curr[:, None] * N + n_offs[None, :],
                     mask=mask_k_curr[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(x, w1)
    
    b1 = tl.load(bias1_ptr + n_offs, mask=mask_n, other=0.0)
    hidden = acc + b1
    
    # GELU approximation
    hidden = 0.5 * hidden * (1.0 + tl.tanh(0.79788456 * (hidden + 0.044715 * hidden * hidden)))
    
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_SIZE_K):
        k_offs_curr = k + k_offs
        mask_k_curr = k_offs_curr < N
        
        h = tl.load(hidden + m_offs[:, None] * N + k_offs_curr[None, :],
                    mask=mask_m[:, None] & mask_k_curr[None, :], other=0.0)
        w2 =
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 28
```
