# Bug: AITER unified attention block_size is hardcoded in platform config instead of reported by backend

## Symptom

When using the AITER unified attention backend on ROCm, the KV cache block size is hardcoded in `RocmPlatform.check_and_update_config()` based on environment variables (set to 64 or 16). This causes a mismatch when the backend internally expects a specific block size (64) but the platform overrides it to a different value (e.g., 16).

The result is incorrect attention computation or crashes due to block size mismatch between the cache layout and the attention kernel.

## Affected Files

- `vllm/platforms/rocm.py`
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

## How to Reproduce

1. Configure vLLM to use the AITER unified attention backend on a ROCm GPU.
2. Set environment variables that cause `check_and_update_config` to choose a block size different from what the backend expects (e.g., block_size=16 when backend needs 64).
3. Run inference — the cache block layout won't match the kernel's expectations.

## Environment

- vLLM at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
