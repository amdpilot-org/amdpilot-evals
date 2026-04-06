# Bug: Hardcoded stride assumptions in Triton kernel `cp_mha_gather_cache_kernel`

## Symptom

The Triton kernel `cp_mha_gather_cache_kernel` in `vllm/v1/attention/backends/rocm_aiter_fa.py` computes KV cache strides using hardcoded arithmetic expressions (triple products of `num_heads * head_size * PAGE_SIZE`) rather than reading the actual strides from the KV cache tensors.

When the KV cache tensor has a non-contiguous memory layout, these hardcoded stride calculations produce incorrect values. This leads to the kernel reading from wrong memory offsets, resulting in incorrect attention outputs or runtime crashes.

## Affected file

- `vllm/v1/attention/backends/rocm_aiter_fa.py`

## Expected behavior

The kernel should correctly handle KV cache tensors regardless of their memory layout (contiguous or non-contiguous) by using the actual tensor strides rather than assuming a specific contiguous layout.
