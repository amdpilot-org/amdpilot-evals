# KernelBench Level 3 Problem 32: 32_ConvolutionalVisionTransformer.py

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
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Convolutional Vision Transformer (CViT) implementation.
        :param num_classes: Number of output classes for classification.
        :param embed_dim: Dimensionality of the embedding space.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
        :param patch_size: Size of the convolutional patches.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        :param image_size: Height/width of the square input image.
        """
        super(Model, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2  # Total number of patches after conv
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the CViT model.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of shape (B, num_classes)
        """
        B = x.size(0)
        x = self.conv1(x)                  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(start_dim=1)         # (B, embed_dim * num_patches)
        x = self.linear_proj(x)            # (B, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 2, embed_dim)

        for layer in self.transformer_layers:
            x = layer(x)

        return self.fc_out(x[:, 0])        # Use [CLS] token for classification

# === Test config ===
batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]

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

A previous single-shot attempt failed. Error: `failed to translate module to LLVM IR`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    stride_x,
    stride_out,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * stride_x
    out_ptr += row_idx * stride_out

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = x_centered * rstd

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm * weight

    if bias_ptr is not None:
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm + bias

    tl.store(out_ptr + cols, x_norm, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    N = x.shape[-1]
    M = x.numel() // N

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)

    layer_norm_kernel[grid](
        x, out, weight, bias, x.stride(0), out.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def fused_mlp_kernel(
    x_ptr,
    out_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    stride_x,
    stride_out,
    stride_w1,
    stride_w2,
    M,
    N,
    K,
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

    m_mask = m_offs < M
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = (k + k_offs) < K
        x_ptrs = x_ptr + m_offs[:, None] * stride_x + (k + k_offs)[None,
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 32
```
