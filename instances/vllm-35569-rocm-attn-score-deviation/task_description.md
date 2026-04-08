# vLLM Issue #35569: ROCM_ATTN Backend Score Deviation

## Bug Description
The ROCM_ATTN attention backend produces ~8.5% systematic score deviation when running the Qwen3-VL-Reranker-2B model for text scoring/pooling on AMD MI355X GPU. The expected score for a reference query pair is 0.100404, but the ROCM_ATTN backend produces a score that deviates by ~8.5%.

## What You Need To Do

1. **Run the test harness** to confirm the issue:
   ```bash
   /usr/bin/python3 /workspace/test_harness.py
   ```

2. **Investigate the ROCM_ATTN backend** implementation in `vllm/attention/backends/rocm_flash_attn.py` and related files.

3. **Find and fix the root cause** of the score deviation. The fix should be in the vLLM source code under `/workspace/vllm/`.

4. **Verify your fix** by running the test harness again and confirming SCORE: 100.

## Important Notes

- The test harness starts a vLLM server with the `--attention-backend ROCM_ATTN` flag and sends scoring requests to `/v1/score`.
- If the server starts but `/v1/score` returns 404, you may need to investigate why the scoring endpoint is not enabled for this model type.
- The model uses `Qwen3VLForConditionalGeneration` architecture with `num_labels=2`.
- This is a reranker/pooling model that uses encoder attention paths.
- Do NOT modify the test harness (`/workspace/test_harness.py`).

## Safe Process Management
When stopping server processes, use targeted kill commands:
```bash
# Safe: kill specific server process by PID
kill -9 <specific_pid>

# UNSAFE — DO NOT USE:
# pkill -f python  # This will kill your own shell!
# kill -9 $(pgrep -f vllm)  # Same problem
```

## Environment
- GPU: AMD MI355X (gfx950)
- ROCm: 7.0
- Model: Qwen/Qwen3-VL-Reranker-2B (cached at /root/.cache/huggingface)
- Python: /usr/bin/python3
