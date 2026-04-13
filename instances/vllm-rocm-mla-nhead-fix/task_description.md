# Bug: MLA decode backend crashes for certain model configurations on ROCm

## Symptom

Certain model configurations crash when using the MLA decode backend on ROCm. The crash occurs during decode buffer allocation, producing a kernel launch failure or illegal memory access.

## How to reproduce

Serve a model with high tensor parallelism through the MLA decode path on a ROCm GPU. The crash occurs during decode initialization.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
