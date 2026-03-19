# Learned Insights

- **Trial 1**: The aiter ASM MLA decode kernel requires valid q_scale and kv_scale tensors when Q is FP8; passing None from layer.k_scale triggers an assertion at asm_mla.cu:206
- **Trial 1**: The fix pattern is: k_scale = layer.k_scale if layer.k_scale is not None else self.k_scale, where self.k_scale = torch.tensor([1.0])
- **Trial 1**: There are exactly 4 mla_decode_fwd call sites in aiter_backend.py: forward_extend target_verify, forward_extend draft_extend non-graph, forward_extend draft_extend graph, and forward_decode
- **Trial 1**: This fallback pattern matches what flashmla_backend.py already does
