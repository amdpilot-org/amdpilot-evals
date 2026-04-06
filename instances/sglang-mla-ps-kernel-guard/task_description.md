# Non-MLA models crash with AttributeError in aiter_backend.py

When running non-MLA models (e.g., Llama, Mistral) with the AITer attention backend on ROCm, the forward pass crashes with:

```
AttributeError: 'AiterAttnBackend' object has no attribute 'max_split_per_batch'
```

The crash occurs during CUDA graph capture and replay. An attribute that is only initialized for MLA models is being accessed unconditionally for all model types.

This breaks ALL non-MLA model inference when using the AITer backend.

## Environment

- SGLang at `/workspace/sglang`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
