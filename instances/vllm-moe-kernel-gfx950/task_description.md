# MoE Kernel Test Failures on AMD MI355X (gfx950)

## Problem

Multiple MoE (Mixture-of-Experts) kernel tests fail on AMD MI355X (gfx950) GPUs. The failures span several independent issues across the MoE subsystem:

1. **FP8 quantization precision errors**: FP8 quantization kernels produce 1-ULP boundary errors causing values to land in wrong FP8 buckets. This affects numerical test assertions.

2. **Platform-gated kernel silently no-ops**: A C++ kernel function compiled with `#ifndef USE_ROCM` becomes a no-op on ROCm, but code paths still attempt to call it, producing incorrect results or crashes.

3. **Test with uninitialized weights**: A MoE test compares against uninitialized (all-zero) weights, making it pass vacuously regardless of correctness — a false-positive that masks real bugs.

4. **API type mismatch**: A `topk` function returns a tuple instead of the expected `SparseMatrix` type on the ROCm platform, causing attribute errors.

These issues collectively prevent the MoE test suite from providing meaningful coverage on gfx950 hardware.

## Reproduction

Run the MoE kernel test suite:
```bash
python -m pytest tests/kernels/moe/ -v --tb=short 2>&1 | grep -E "FAILED|ERROR"
```

Individual failing test files include tests for GPT-OSS Triton kernels, modular kernel combinations, routing, shared fused MoE routed transforms, and SiLU+Mul+FP8 quant with DeepGemm.

## Environment

- vLLM at `/workspace/vllm`
- Docker container with ROCm 7.2, PyTorch, AMD MI355X GPUs
- Use `/usr/bin/python3` for all commands
- Single GPU sufficient for kernel-level tests

## Verification

Run the test harness after applying your fix:
```bash
/usr/bin/python3 /workspace/test_harness.py
```

The harness runs targeted MoE kernel correctness checks and verifies that platform-specific code paths are handled correctly on ROCm. A score of 100.0 means all checks pass.
