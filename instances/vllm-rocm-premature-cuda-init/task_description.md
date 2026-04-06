# ROCm: premature CUDA initialization from vllm.platforms import

Importing `vllm.platforms` triggers a call to `torch.cuda.get_device_properties()` at module load time, which initializes the CUDA runtime before Ray workers have a chance to set `CUDA_VISIBLE_DEVICES`. As a result, every tensor-parallel worker sees all GPUs and defaults to GPU 0 instead of being pinned to its assigned device. Multi-GPU tensor-parallel inference silently runs all workers on the same GPU, producing incorrect results or crashes.

## Environment

- vLLM repo at `/workspace/vllm`
- Use `/opt/venv/bin/python3`

## Verification

```bash
/opt/venv/bin/python3 /workspace/test_harness.py
```
