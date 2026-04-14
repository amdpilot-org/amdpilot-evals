# vLLM Hybrid Model KV Cache Stride Corruption on ROCm

## Bug Description
When running hybrid attention/Mamba models on ROCm GPUs, the attention output contains garbage values and NaN. The issue is in the KV cache gather kernel used by the ROCm aiter attention backend.

Hybrid models use an interleaved KV block memory layout (`[K_0][V_0][K_1][V_1]...`) instead of the standard contiguous layout (`[K_all][V_all]`). The Triton kernel that gathers cached KV entries uses pointer arithmetic that assumes the contiguous layout, causing it to read from incorrect memory locations when the hybrid interleaved layout is active.

The result is silent output corruption — no crash, but attention outputs contain garbage values and NaN, producing incoherent or corrupted model responses.

## What You Need To Do

1. **Run the test harness** to confirm the issue:
   ```bash
   /usr/bin/python3 /workspace/test_harness.py
   ```

2. **Investigate the KV cache gather kernel** and the hybrid attention memory layout handling in the ROCm aiter attention backend.

3. **Find and fix the root cause** — ensure the kernel uses correct stride/pointer arithmetic for the interleaved KV block layout used by hybrid models.

4. **Verify your fix** by running the test harness again and confirming SCORE improves.

## Important Notes

- This is a **silent correctness bug** — there is no crash or error message. The model just produces wrong outputs.
- The bug only affects hybrid models (models that mix attention and Mamba/SSM layers). Pure attention models are unaffected.
- The fix is in the vLLM source code under `/workspace/vllm/`.
- Do NOT modify the test harness (`/workspace/test_harness.py`).

## Environment
- GPU: AMD MI355X (gfx950)
- ROCm: 7.2
- Framework: vLLM
- Python: /usr/bin/python3
