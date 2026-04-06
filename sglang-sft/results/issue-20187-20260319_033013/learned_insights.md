# Learned Insights

- **Trial 1**: SGLang's aiter_backend.py forward_extend has two major branches: a non-radix path (extend_no_prefix) and a radix-cache path (elif layer.qk_head_dim != kv_lora_rank+qk_rope_head_dim). FP8 prefill support must be added to both independently.
- **Trial 1**: fused_gemm_afp4wfp4_split_cat from aiter.ops.triton.gemm.fused_gemm_afp4wfp4 fuses the GEMM + split + k_pe concatenation into a single operation producing FP8 k/v directly, avoiding extra element-wise casts that would otherwise be needed.
- **Trial 1**: dynamic_mxfp4_quant from aiter.ops.triton.quant is used to quantize cached latent vectors before passing them to the fused GEMM.
- **Trial 1**: The FP8 prefill path gates on _use_fp8_prefill_attn flag AND checks that kv_b_proj weights are uint8 dtype (MXFP4), then uses mla_prefill_ps_asm_fwd + mla_reduce_v1 for FP8 attention instead of flash_attn_varlen_func.
