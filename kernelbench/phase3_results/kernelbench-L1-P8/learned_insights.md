# Learned Insights

- **Trial 1**: AMD MI355X (gfx950, CDNA4) MFMA instructions require FP16 or BF16 inputs for tl.dot — FP32 inputs cause LLVM unrealized_conversion_cast error
- **Trial 1**: Working Triton matmul pattern for ROCm: Load FP32 -> cast to FP16 -> tl.dot with FP32 accumulator -> store FP32
- **Trial 1**: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64 gives 1.08ms for M=8205,K=2949,N=5921 matmul; larger 256x256 blocks regress to 2.33ms
- **Trial 1**: Baseline PyTorch torch.matmul for this problem (M=8205,K=2949,N=5921) takes ~2.0ms on MI355X
- **Trial 2**: Trial 2 produced no output — agent may have gotten stuck on planning without executing. Need explicit instructions to run benchmark first.
- **Trial 2**: For M=8205,K=2949,N=5921 with BLOCK 128x128x64: boundary waste is 13 elements on M, 97 on N, 5 on K — suggests room for block size tuning
- **Trial 2**: L2 cache tile swizzling (GROUP_SIZE_M pattern) is a standard Triton GEMM optimization that should be tried early
- **Trial 3**: Agent failed 2 consecutive trials with no output on stage2 - needs extremely explicit step-by-step instructions with exact commands
- **Trial 3**: For M=8205,K=2949,N=5921: BLOCK_N=64 gives waste of 33 elements vs BLOCK_N=128 giving waste of 97 elements on N dimension
- **Trial 3**: L2 cache tile swizzling (GROUP_SIZE_M) and num_warps/num_stages tuning are untried optimizations for this kernel
- **Trial 4**: Agent has failed 3 consecutive trials (2,3,4) with no output - needs extremely explicit copy-paste commands with zero planning phase
- **Trial 4**: Working baseline: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, score=68.60 (1.08ms) - file is /workspace/generated_kernel.py
- **Trial 4**: Untried optimizations: L2 tile swizzling (GROUP_SIZE_M), num_warps tuning, num_stages tuning, BLOCK_N=64
- **Trial 5**: Agent has failed 4 consecutive trials (2-5) with no output on optimization stages - needs copy-paste ready commands with zero planning
- **Trial 5**: Working kernel at /workspace/generated_kernel.py achieves score 68.60 (1.08ms) with BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, FP16 dot
- **Trial 5**: L2 tile swizzling (GROUP_SIZE_M), num_warps tuning, num_stages tuning, BLOCK_N=64 remain untried optimizations
