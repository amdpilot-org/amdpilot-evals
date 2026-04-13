# vLLM ROCm: encoder-decoder models fail on AMD GPUs

Encoder-decoder models (e.g., Whisper, BART) fail when running on ROCm GPUs. The error occurs during attention computation when the model attempts to process encoder self-attention sequences.

## How to reproduce

Attempt to run an encoder-decoder model on a ROCm GPU using vLLM's v1 attention path. The model fails during the attention forward pass.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
