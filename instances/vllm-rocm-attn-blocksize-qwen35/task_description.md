# ROCm attention: Qwen3.5-style KV block sizes break Triton fallback output

Models such as **Qwen3.5** use a KV block size of **1056**, which is a multiple of 16 but **not** a power of two. On the ROCm attention backend, the Triton fallback path effectively only allows **power-of-two** block sizes. With a non-power-of-two, valid-looking block size (1056, or other multiples of 16 like 48 or 80), generation becomes corrupted—for example **nonsensical repetition** (e.g. runs of punctuation like `!!!!!!`) instead of coherent text.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
