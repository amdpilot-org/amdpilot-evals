# [AMD] Fix FP8 assertion failure in aiter MLA decode by falling back to self.k_scale

## Bug Description

When `layer.k_scale` is `None` (the default in `RadixAttention`), the aiter ASM MLA kernel asserts `q_scale.has_value() && kv_scale.has_value()` for FP8 Q tensors. This assertion failure occurs in `aiter/csrc/py_itfs_cu/asm_mla.cu:206` when running MLA models (e.g., Kimi-K2.5, DeepSeek-R1) with `--attention-backend aiter --kv-cache-dtype fp8_e4m3`.

This issue was found in Kimi-K2.5 but not in DeepSeek-R1. It must be resolved to enable the FP8 KV cache for Kimi-K2.5 on MI355.

## Root Cause

The aiter ASM MLA decode kernel requires valid `q_scale` and `kv_scale` tensors when Q is FP8, but all 4 `mla_decode_fwd` call sites in `aiter_backend.py` pass `layer.k_scale` directly, which defaults to `None` in `RadixAttention`. The `None` propagates as an empty `std::optional` to the C++ kernel, triggering the assertion during CUDA graph capture.

## Expected Fix

Fall back to `self.k_scale` (initialized to `torch.tensor([1.0])` at `AiterAttnBackend.__init__`) when `layer.k_scale is None`, at all 4 call sites:

1. `forward_extend` → target_verify path
2. `forward_extend` → draft_extend non-graph path
3. `forward_extend` → draft_extend graph path
4. `forward_decode`

This matches the fallback pattern already used by `flashmla_backend.py`.

## Environment

- **Hardware**: MI355 x8
- **Repository**: SGLang (sgl-project/sglang)
- **Codebase**: SGLang is at `/sgl-workspace/sglang` (checked out at a version BEFORE this fix)
- **Key file**: `python/sglang/srt/layers/attention/aiter_backend.py`
- **Model**: Kimi-K2.5 at `/models/Kimi-K2.5`

## Reproduction

Start the server:
```bash
SGLANG_AITER_MLA_PERSIST=1 /opt/venv/bin/python3 -m sglang.launch_server \
  --model-path /models/Kimi-K2.5 \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --chunked-prefill-size 131072 \
  --host 0.0.0.0 \
  --port 9000 \
  --log-requests \
  --disable-radix-cache \
  --mem-fraction-static 0.8 \
  --max-running-requests 64 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend aiter
```

The server will crash with an assertion error in `asm_mla.cu:206`.

## Verification

After fixing, the server should start successfully and the GSM8K benchmark should run:
```bash
/opt/venv/bin/python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
  --num-questions 50 --parallel 10 --num-shots 5 --port 9000
```

The test harness at `/workspace/test_harness.py` will:
1. Start the server with the FP8 KV cache + aiter backend
2. Send a health check / warmup request
3. Report SCORE: 100 if the server starts and responds correctly
