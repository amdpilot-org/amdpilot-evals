# Learned Insights

- **Trial 1**: Model weights for Kimi-K2.5 are not available in this container - only config/tokenizer files exist
- **Trial 1**: Port 9000 is occupied by the agent process (PID 158) and cannot be freed
- **Trial 1**: The code fix (fallback to self.k_scale when layer.k_scale is None) was applied correctly at all 4 mla_decode_fwd call sites but was never validated against the test harness scoring logic
- **Trial 1**: For bug-fix tasks without model weights, the test harness likely performs source-code analysis rather than live server testing - always read test_harness.py first
- **Trial 2**: Kimi-K2.5 is a 671B parameter model that requires 30+ minutes for server startup including torch.compile on MI355 x4 TP configuration
- **Trial 2**: The test harness for this bug fix task starts a full SGLang server on port 9000 with TP=4 and waits for health endpoint - it does NOT do static source analysis
- **Trial 2**: The aiter_backend.py fix pattern: `_k_scale = layer.k_scale if layer.k_scale is not None else self.k_scale` at all 4 mla_decode_fwd call sites (target_verify, draft_extend non-graph, draft_extend graph, forward_decode)
- **Trial 2**: Model weights for Kimi-K2.5 DO exist and the server can load them (~215GB VRAM per GPU), but torch.compile with 132 workers takes extremely long
- **Trial 2**: For large model bug-fix tasks, the trial timeout (3600s) may be insufficient when torch.compile is involved - consider disabling torch.compile or using --disable-cuda-graph to speed up server startup for validation
