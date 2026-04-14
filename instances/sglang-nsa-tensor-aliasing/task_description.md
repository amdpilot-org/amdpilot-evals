# SGLang NSA Attention Tensor Aliasing Crash on ROCm

## Bug Description
When running models with NSA (Native Sparse Attention) on ROCm GPUs, the inference crashes with a tensor memory aliasing error. The crash occurs in the NSA indexer and MHA forward paths when tensor write-back operations attempt to write to memory that overlaps with the source tensor.

The error manifests as:
```
RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to the same memory location
```

This crash does not occur on CUDA because CUDA's PyTorch backend does not enforce the aliasing check.

## What You Need To Do

1. **Run the test harness** to confirm the issue:
   ```bash
   /opt/venv/bin/python3 /workspace/test_harness.py
   ```

2. **Investigate the NSA attention code paths** in `python/sglang/srt/layers/attention/nsa/` and related model forward methods.

3. **Find and fix the root cause** of the tensor aliasing violations. The fix should ensure that write-back operations do not write to memory locations that overlap with the source tensor.

4. **Verify your fix** by running the test harness again and confirming SCORE improves.

## Important Notes

- The crash occurs specifically on ROCm because ROCm's PyTorch enforces tensor aliasing checks that CUDA silently allows.
- Multiple code paths may be affected — ensure all aliasing violations are fixed.
- Do NOT modify the test harness (`/workspace/test_harness.py`).

## Environment
- GPU: AMD MI355X (gfx950)
- ROCm: 7.2
- Framework: SGLang
- Python: /opt/venv/bin/python3
