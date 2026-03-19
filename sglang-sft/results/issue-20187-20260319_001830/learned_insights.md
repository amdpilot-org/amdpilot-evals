# Learned Insights

- **Trial 1**: For SGLang FP8 prefill with radix cache, kv_indptr (not qo_indptr) must be used for FP8 metadata since KV includes prefix tokens
- **Trial 1**: fused_gemm_afp4wfp4_split_cat is invoked via layer.kv_b_proj() with a tuple argument containing fp8_dtype output type
- **Trial 1**: Extracting shared FP8 prefill logic into a helper method (mla_fp8_prefill_attn) allows both prefix and no-prefix paths to reuse the same code
- **Trial 1**: total_s must use seq_lens_sum to cover full KV including prefix for radix-cache path
