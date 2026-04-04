# Task: ROCm attention with non-power-of-two KV block size

## Symptom

The model **Qwen3-Next-80B-A3B-Thinking** uses a KV cache **block size of 544**, which is **not** a power of two.

On **ROCm**, when using the **ROCm attention** backend (`rocm_attn`), inference **crashes or misbehaves** during attention / KV-cache work. The failure is tied to the **Triton** attention and cache paths: they effectively assume **power-of-two** block sizes and use **bitwise** addressing patterns that are invalid for a size like **544**.

You need to make the ROCm attention stack **correctly support** this non-standard physical block size end-to-end (cache reshape/write and paged / prefix attention kernels), without breaking existing power-of-two block sizes.
