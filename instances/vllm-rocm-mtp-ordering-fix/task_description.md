# vLLM MTP Speculative Decoding Crash with Aiter Sparse MLA on ROCm

## Bug Description
When running DeepSeek-V3.2 with MTP (Multi-Token Prediction) speculative decoding and `num_speculative_tokens > 1` on ROCm with the `ROCM_AITER_MLA_SPARSE` attention backend, the server crashes with:

```
ValueError: Unsupported attention metadata type for speculative decoding
```

The crash occurs in the speculative decode proposer when validating attention metadata types. The validation depends on dictionary iteration order, which causes it to check the wrong metadata type for certain model architectures.

The issue does not occur with `num_speculative_tokens=1` or with other attention backends.

## What You Need To Do

1. **Run the test harness** to confirm the issue:
   ```bash
   /usr/bin/python3 /workspace/test_harness.py
   ```

2. **Investigate the speculative decode proposer** in `vllm/v1/spec_decode/` and the attention metadata type validation logic.

3. **Find and fix the root cause** — the validation should not depend on dictionary iteration order.

4. **Verify your fix** by running the test harness again and confirming SCORE improves.

## Important Notes

- The error message names the unsupported type but does NOT reveal why the validation fails (ordering dependency).
- The fix may require changes in multiple files within the spec_decode directory.
- Do NOT modify the test harness (`/workspace/test_harness.py`).

## Environment
- GPU: AMD MI355X (gfx950)
- ROCm: 7.2
- Framework: vLLM
- Python: /usr/bin/python3
