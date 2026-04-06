# Learned Insights

- **Trial 1**: In SGLang's aiter_backend.py, layer.k_scale defaults to None in RadixAttention. The aiter ASM MLA kernel (asm_mla.cu:206) asserts q_scale and kv_scale are non-empty when Q is FP8. The fix is to fall back to self.k_scale (initialized as torch.tensor([1.0])) at all 4 mla_decode_fwd call sites.
- **Trial 1**: The fallback pattern matches flashmla_backend.py's existing approach — always check flashmla_backend.py as a reference for how scale parameters are handled.
- **Trial 1**: The 4 mla_decode_fwd call sites in aiter_backend.py are: forward_extend target_verify, forward_extend draft_extend non-graph, forward_extend draft_extend graph, and forward_decode.
