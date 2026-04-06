# KernelBench Level 3 Problem 47: 47_NetVladNoGhostClusters.py

## Goal

Write an optimized Triton kernel implementation (`ModelNew`) that:
1. Produces the **exact same output** as the PyTorch reference `Model`
2. Is **faster** than the PyTorch baseline
3. Uses Triton `@triton.jit` kernels (NOT raw CUDA/HIP)

## PyTorch Reference Implementation

```python
# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code modified from here
https://github.com/albanie/collaborative-experts/blob/master/model/net_vlad.py
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class Model(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(Model, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK

batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
  return [torch.rand(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]

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

A previous single-shot attempt failed. The output was incorrect.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_softmax_bn_kernel(
    x_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # First pass: apply batch norm and find max for softmax
    max_val = -1e10
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-1e10)
        
        # Apply batch norm
        bn_w = tl.load(bn_weight_ptr + cols, mask=mask, other=1.0)
        bn_b = tl.load(bn_bias_ptr + cols, mask=mask, other=0.0)
        bn_m = tl.load(bn_mean_ptr + cols, mask=mask, other=0.0)
        bn_v = tl.load(bn_var_ptr + cols, mask=mask, other=1.0)
        
        x_bn = (x - bn_m) * bn_w / tl.sqrt(bn_v + 1e-5) + bn_b
        max_val = tl.maximum(max_val, tl.max(x_bn, axis=0))
    
    # Second pass: compute exp and sum
    sum_exp = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        
        # Apply batch norm
        bn_w = tl.load(bn_weight_ptr + cols, mask=mask, other=1.0)
        bn_b = tl.load(bn_bias_ptr + cols, mask=mask, other=0.0)
        bn_m = tl.load(bn_mean_ptr + cols, mask=mask, other=0.0)
        bn_v = tl.load(bn_var_ptr + cols, mask=mask, other=1.0)
        
        x_bn = (x - bn_m) * bn_w / tl.sqrt(bn_v + 1e-5) + bn_b
        exp_x = tl.exp(x_bn - max_val)
        sum_exp += tl.sum(exp_x, axis=0)
    
    # Third pass: normalize and store
    for i in range(0, n_cols, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        
        # Apply batch norm
        bn_w = tl.load(bn_weight_ptr + cols, mask=mask, other=1.0)
        bn_b = tl.load(bn_bias_ptr + cols, mask=mask, other=0.0)
        bn_m = tl.load(bn_mean_ptr + cols, mask=mask, other=0.0)
        bn_v = tl.load(bn_var_ptr + cols, mask=mask, other=1.0)
        
        x_bn = (x - bn_m) * bn_w / tl.sqrt(bn_v + 1e-5) + bn_b
        exp_x = tl.exp(x_bn - max_val)
        out = exp_x / (sum_exp + 1e-10)
        tl.store(out_ptr + row_start + cols, out, mask=mask)


@triton.jit
def vlad_aggregation_kernel(
    assignment_ptr,
    x_ptr,
    clusters2_ptr,
    a_sum_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_assign_b,
    stride_assign_n,
    stride_assign_k,
    stride_x_b,
    stride_x_n,
    stride_x_d,
    stride_out_b,
    stride_out_d,
    stride_out_k,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batc
```

Analyze what went wrong and fix the issues.

## Output Requirements

Save your implementation to `/workspace/generated_kernel.py`.
The file must define a `ModelNew` class with the same interface as `Model`.
Run the test harness to verify:
```bash
/opt/venv/bin/python3 /workspace/test_harness.py --level 3 --problem-id 47
```
