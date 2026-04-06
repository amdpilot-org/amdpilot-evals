# Learned Insights

- **Trial 1**: ROCm Triton does not support tl.libdevice.tanh - must use manual implementation with tl.math.exp for tanh
- **Trial 1**: Triton LayerNorm kernel must use tl.where to zero out masked elements (beyond n_cols when BLOCK_SIZE > n_cols) before variance calculation, otherwise (-mean)^2 terms corrupt variance
- **Trial 1**: BLOCK_SIZE must be power of 2 in Triton kernels
- **Trial 1**: MiniGPTBlock profiling breakdown: attention_matmul ~40%, layernorm ~25%, linear_projections ~20%, gelu ~15%
- **Trial 1**: Triton LayerNorm kernel saved ~1.5ms, Triton GELU kernel saved ~0.8ms on MiniGPTBlock
- **Trial 1**: Fused QKV kernel attempt failed due to indexing complexity - consider simpler fusion targets first like residual+LayerNorm or linear+GELU
- **Trial 1**: KernelBench score 62.40 corresponds to 9.96ms runtime vs 12.3ms reference (1.235x speedup)
- **Trial 2**: Agent produced no output in trial 2 — likely crashed or timed out before any work was done
- **Trial 2**: torch.nn.functional.scaled_dot_product_attention with is_causal=True is the recommended replacement for manual attention implementation on ROCm — uses flash attention backend
- **Trial 2**: Fusing residual+LayerNorm and linear+GELU are lower-risk optimizations than full fused attention kernel
- **Trial 3**: Agent crashed/timed out in trials 2 and 3 with no output — likely attempting too-complex changes without incremental testing
- **Trial 3**: torch.nn.functional.scaled_dot_product_attention with is_causal=True is the highest-leverage single optimization for the 40% attention bottleneck
- **Trial 4**: Agent crashed/timed out 3 consecutive trials (2,3,4) with no output — complex multi-kernel changes cause silent failures
- **Trial 4**: When agent repeatedly crashes, give single-change instructions with immediate test checkpoint
- **Trial 4**: The working baseline at score 62.40 is in /workspace/generated_kernel.py with Triton LayerNorm + GELU kernels
- **Trial 5**: Agent has crashed 4 consecutive trials (2-5) attempting complex multi-kernel changes — ultra-minimal single-change instructions are required
- **Trial 5**: The working generated_kernel.py at score 62.40 must be preserved as fallback — never rewrite from scratch
- **Trial 5**: torch.nn.functional.scaled_dot_product_attention(is_causal=True) is the single highest-leverage optimization for the 40% attention bottleneck
