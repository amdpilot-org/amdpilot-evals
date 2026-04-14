# ROCm: premature CUDA initialization during platform detection

Importing `vllm.platforms` initializes the CUDA runtime at module load time, before Ray workers have a chance to set `CUDA_VISIBLE_DEVICES`. As a result, every tensor-parallel worker sees all GPUs and defaults to GPU 0 instead of being pinned to its assigned device. Multi-GPU tensor-parallel inference silently runs all workers on the same GPU, producing incorrect results or crashes.

## How to reproduce

Import `vllm.platforms` in a fresh Python process and check whether `torch.cuda.is_initialized()` returns True afterward. It should not.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
