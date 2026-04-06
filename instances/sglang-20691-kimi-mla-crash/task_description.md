# Fix Kimi K2 crash on ROCm with fused decode MLA

## Issue

GitHub Issue: https://github.com/sgl-project/sglang/issues/20691

Kimi K2 (DeepSeek V2 architecture) crashes on ROCm during CUDA graph capture when using the default fused decode MLA code path. The crash occurs because the ROCm fused MLA rope code in `forward_mla_fused_rope_rocm.py` tries to access `forward_metadata` on a `HybridAttnBackend` object, but `HybridAttnBackend` does not have that attribute (it has `init_forward_metadata` instead).

### Error 1 — AttributeError
```
File "forward_mla_fused_rope_rocm.py", line 111, in forward_absorb_fused_mla_rope_prepare
    forward_batch.attn_backend.forward_metadata
AttributeError: 'HybridAttnBackend' object has no attribute 'forward_metadata'. Did you mean: 'init_forward_metadata'?
```

### Error 2 — TypeError (related)
```
File "forward_mla_fused_rope_rocm.py", line 118, in forward_absorb_fused_mla_rope_prepare
    attn_logits, _, kv_indptr, kv_indices, _, _, _ = attn_backend.forward_metadata
TypeError: cannot unpack non-iterable ForwardMetadata
```

### Workaround
Setting `SGLANG_ROCM_FUSED_DECODE_MLA=0` avoids the crash, but users shouldn't need to do this.

### Suspected root cause
This was likely introduced by PR #19122 which changed attention backend interfaces. The ROCm fused MLA path needs to be updated to work with the current `HybridAttnBackend` API.

## Reproduction Command
```bash
sglang serve --model-path fxmarty/moonshotai_Kimi-K2-Instruct-0905-2-layers \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --decode-attention-backend triton \
  --prefill-attention-backend aiter
```

## Key Files to Investigate
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` — where the crash happens
- `python/sglang/srt/layers/attention/aiter_backend.py` — HybridAttnBackend definition
- `python/sglang/srt/models/deepseek_v2.py` — model forward path

## Environment

- GPU: AMD Instinct MI355X
- ROCm 7.2
- SGLang source: `/sgl-workspace/sglang/` (editable install — edit files here)
- Workspace symlink: `/workspace/sglang` → `/sgl-workspace/sglang`
- Use `/opt/venv/bin/python3` for all commands
- Model: `fxmarty/moonshotai_Kimi-K2-Instruct-0905-2-layers` (a tiny 2-layer variant for testing)

## Rules

- Edit files ONLY under `/sgl-workspace/sglang/`
- Do NOT run `pip install -e .` — this would overwrite ROCm PyTorch
- Use `/opt/venv/bin/python3` for all Python commands
- Do NOT modify test harness or benchmark scripts
- Kill leftover server processes with safe pattern:
  `pgrep -f "python3 -m sglang" | xargs -r kill -9`
- Make clean, minimal fixes only — no debug prints

## Test Harness and Server Startup

The test harness starts a model server with TP=8. Model loading takes several minutes even for this small model. **Make the fix FIRST based on code analysis, then run the test harness ONCE to verify.** Do NOT repeatedly restart the server.

## Verification

After applying your fix, run:
```bash
pgrep -f "python3 -m sglang" | xargs -r kill -9; sleep 2
/opt/venv/bin/python3 /workspace/test_harness.py
```

A score of 100 means the server starts, serves requests, and produces no crash errors.
