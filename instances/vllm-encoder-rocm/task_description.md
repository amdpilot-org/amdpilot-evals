# Enable Encoder Self-Attention on ROCm Attention Backends

## Problem

vLLM's ROCm attention backends (`RocmAttentionImpl` and `RocmAiterUnifiedAttention`) raise `NotImplementedError` for encoder self-attention (`AttentionType.ENCODER` and `ENCODER_DECODER`). This blocks Whisper and other encoder-decoder models on AMD GPUs.

## Affected Files

- `vllm/v1/attention/backends/rocm_attn.py`
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

## Environment

- Use `/opt/venv/bin/python3` for all commands

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
