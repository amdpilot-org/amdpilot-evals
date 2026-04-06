# Learned Insights

- **Trial 1**: RegNet Level 3 Problem 27: Conv2d accounts for ~70% of runtime, BN+ReLU ~25%, GAP+FC ~5%
- **Trial 1**: torch.compile conflicts with custom Triton kernels on AMD ROCm — MLIR pass failures in ConvertTritonAMDGPUToLLVM. Do NOT compile regions containing @triton.jit calls.
- **Trial 1**: Training-mode BatchNorm in Triton requires biased variance (divide by N, not N-1) and running stats updates — very complex to get right
- **Trial 1**: Fused GAP+FC Triton kernel was 29% slower than separate ops due to nested loop overhead
- **Trial 1**: cudnn.benchmark=True and channels_last memory format did not provide measurable speedup for this RegNet workload
- **Trial 1**: Baseline for RegNet is ~1.72ms with score 60 (1.0x speedup). The Triton GAP kernel matches but does not beat baseline.
- **Trial 2**: For complex models like RegNet where Conv2d dominates (70%), custom Triton kernels cannot beat MIOpen/cuDNN. Focus on PyTorch-level optimizations: torch.compile, BN folding, channels_last format.
- **Trial 2**: torch.compile MUST be used WITHOUT any @triton.jit kernels on AMD ROCm to avoid MLIR pass failures.
- **Trial 2**: BN folding (merging BatchNorm weights into Conv2d weights) can eliminate ~25% of runtime for inference-like workloads.
- **Trial 3**: Trial 3 of RegNet produced no output — agent got stuck. Providing exact code recipes is necessary when time is short.
- **Trial 3**: For RegNet, the viable optimization path is: channels_last memory format + torch.compile on feature_extractor (without Triton kernels in compiled region) + Triton GAP kernel outside compiled region
- **Trial 4**: Agent got stuck with no output for trials 3 and 4 on RegNet — need extremely specific copy-paste code when time is short
- **Trial 4**: Pure PyTorch ModelNew with torch.compile (no @triton.jit) is the untested viable path for RegNet on AMD ROCm
- **Trial 4**: channels_last memory format conversion should be done once in __init__ via self.to(memory_format=torch.channels_last), not per-forward
- **Trial 5**: Agent has been stuck for 3 consecutive trials (3, 4, 5) on RegNet — the simplest viable approach is a pure PyTorch ModelNew matching the reference architecture exactly
- **Trial 5**: When agent produces no output repeatedly, provide the simplest possible solution first before attempting optimizations
