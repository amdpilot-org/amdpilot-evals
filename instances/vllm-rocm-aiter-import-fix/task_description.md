# Bug: AITER backend crashes when VLLM_ROCM_USE_AITER=0 and backend is explicitly selected

## Symptom

On ROCm, setting the environment variable `VLLM_ROCM_USE_AITER=0` to disable the AITER backend by default, then explicitly selecting the AITER backend via `attention_config`, causes a crash:

```
AttributeError: 'builtin_function_or_method' object has no attribute 'flash_attn_varlen_func'
```

The environment variable is intended to control the **default** backend selection (auto-discovery), but it currently also prevents explicit backend selection from working. When a user disables AITER auto-discovery via the env var but explicitly requests the AITER backend in their configuration, the system should honor the explicit request. Instead, the aiter module never gets imported and calls to its functions fail with AttributeError.

The same crash occurs for `pa_fwd_asm` (paged attention assembly kernel).

## Affected files

- `vllm/_aiter_ops.py`
- `vllm/v1/attention/backends/rocm_aiter_fa.py`
- `vllm/v1/spec_decode/eagle.py`

## Expected behavior

- The env var `VLLM_ROCM_USE_AITER=0` should only control whether the AITER backend is chosen during **auto-discovery** (default backend selection).
- Explicitly selecting the AITER backend via `attention_config` should always work as long as the aiter library is physically available on the system, regardless of the env var setting.
- There should be a clear separation between "can aiter work on this system" (platform + library check) and "should aiter be used by default" (adds env var check).
