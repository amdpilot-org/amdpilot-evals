# Learned Insights

- **Trial 1**: KernelBench scoring: score 60 corresponds to ~1x speedup (correct implementation matching baseline)
- **Trial 1**: torch.compile fails on ROCm with MLIR error: 'ttg.async_copy_global_to_local' legalization failure
- **Trial 1**: channels_last memory format causes 2x slowdown on this GoogLeNet problem due to format conversion overhead
- **Trial 1**: MIOpen/rocBLAS convolutions are already near-optimal for this problem - do not try to replace them with Triton
- **Trial 1**: Profiling breakdown for GoogLeNet Inception: GEMM_conv=33.7%, MaxPool=28.2%, Conv_miopen=22.6%, bias=5.9%, cat=5.8%
- **Trial 1**: Triton concat kernels on AMD GPUs are prone to GPU memory access faults due to complex multi-branch indexing
- **Trial 2**: Trial 2 produced no output - agent may have gotten stuck or crashed without any work product
- **Trial 2**: For GoogLeNet Inception, pre-allocating output and using slice assignment can eliminate cat overhead (~5.8% of runtime)
- **Trial 2**: CUDA streams for parallel branch execution is worth trying for Inception-style architectures with independent branches
- **Trial 3**: Agent has crashed/timed out on 2 consecutive trials for GoogLeNet Inception - need extremely concrete code examples
- **Trial 3**: Pre-allocated output with slice assignment eliminates torch.cat overhead (~5.8% of runtime) without risk of memory faults
- **Trial 3**: CUDA streams for parallel Inception branches is the safest optimization that doesn't require replacing vendor conv kernels
- **Trial 4**: Agent has crashed 3 consecutive trials on GoogLeNet Inception - needs complete code snippets not architectural guidance
- **Trial 4**: Score 60 = correct implementation at ~1x baseline speed; any speedup above 1x increases score beyond 60
- **Trial 5**: Agent crashes repeatedly when given architectural guidance without complete code - must provide full copy-paste implementations
- **Trial 5**: For GoogLeNet Inception on AMD, a Triton copy_concat_kernel replacing torch.cat is the safest Triton usage that satisfies the requirement while keeping MIOpen convolutions
