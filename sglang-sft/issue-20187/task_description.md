# [AMD] FP8 prefill integration with radix cache path for DeepSeek models

## Bug Description

FP8 prefill attention on DeepSeek models does not cover the radix-cache path. When radix cache is enabled alongside FP8 prefill (`SGLANG_AITER_FP8_PREFILL_ATTN=1`), the prefill attention falls back to BF16 instead of using FP8, losing the 1.17×-1.26× total throughput improvement that FP8 prefill provides.

## Expected Fix

Enable FP8 prefill attention for the radix-cache path:
1. In the aiter attention backend, add FP8 prefill support in the radix-cache code path
2. To reduce extra element-wise casts, use `fused_gemm_afp4wfp4_split_cat` following the same design principle as the existing FP8 prefill path (non-radix-cache path)
3. The fix should NOT cover fp8 fused gemm yet — only the attention prefill path

## Environment

- **Hardware**: MI355 x8
- **Repository**: SGLang (sgl-project/sglang)
- **Codebase**: SGLang is at `/sgl-workspace/sglang` (checked out at a version BEFORE this fix)
- **Key file**: `python/sglang/srt/layers/attention/aiter_backend.py`
- **Model**: DeepSeek-R1-MXFP4-Preview at `/models/DeepSeek-R1-MXFP4-Preview`

## Reproduction

Start the server WITH radix cache (default) and FP8 prefill:
```bash
SGLANG_AITER_FP8_PREFILL_ATTN=1 /opt/venv/bin/python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-R1-MXFP4-Preview \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --chunked-prefill-size 131072 \
  --host 0.0.0.0 \
  --port 9000 \
  --log-requests \
  --mem-fraction-static 0.8 \
  --max-running-requests 64 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend aiter \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

## Verification

After fixing, the server should start successfully with radix cache + FP8 prefill enabled. The benchmark should run and show FP8 prefill is active:
```bash
/opt/venv/bin/python3 -m sglang.bench_serving \
  --host localhost --port 9000 \
  --model DeepSeek-R1-MXFP4-Preview/ \
  --dataset-name random \
  --random-input 2000 --random-output 200 --random-range-ratio 1.0 \
  --num-prompt 8 --max-concurrency 1
```

Expected improvement with FP8 prefill vs BF16 prefill:
- Total tok/s: 1.17×-1.26× improvement
- TTFT: 1.07×-1.29× improvement

The test harness at `/workspace/test_harness.py` will:
1. Start the server with FP8 prefill + radix cache
2. Send test requests and verify responses
3. Report SCORE: 100 if the server starts and serves correctly with radix cache + FP8 prefill
