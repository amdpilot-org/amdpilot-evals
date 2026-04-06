# Learned Insights

- **Trial 1**: In SGLang's aiter_backend.py, the radix-cache path for FP8 prefill requires using kv_indptr from mla_indices_updater_prefill (not qo_indptr) when calling make_mla_prefill_ps_meta_data
- **Trial 1**: total_s for prefill PS metadata must use forward_batch.seq_lens_sum (which includes prefix tokens) rather than forward_batch.extend_seq_lens.sum() (which only counts new tokens)
- **Trial 1**: FP8 prefill attention uses fused_gemm_afp4wfp4_split_cat triggered via kv_b_proj tuple-dispatch pattern: a 5-tuple (kvc, k_pe, nope_head_dim, v_head_dim, fp8_dtype)
- **Trial 1**: Extracting shared FP8 prefill logic into a mla_fp8_prefill_attn helper method avoids code duplication between no-prefix and radix-cache paths
- **Trial 1**: The test harness checks 6 structural properties: helper method existence, kv_indptr source, total_s computation, tuple-dispatch pattern, radix-cache FP8 branch, and no-prefix refactoring
