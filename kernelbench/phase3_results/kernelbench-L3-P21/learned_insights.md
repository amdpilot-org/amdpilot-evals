# Learned Insights

- **Trial 1**: torch.compile(mode='default') gives 1.32x speedup on EfficientNet MBConv (5.68ms->4.31ms, score 63.2)
- **Trial 1**: Manual Triton conv kernels (pointwise 1x1, depthwise) are extremely complex due to 4D indexing, tl.dot constraints, and shared memory limits — avoid writing them from scratch
- **Trial 1**: Hybrid torch.compile + custom Triton kernels fails on ROCm due to 'cannot import specialize_impl from triton.runtime.jit' — need to call Triton kernels outside the torch.compile region
- **Trial 1**: Profiling breakdown for MBConv: depthwise_conv=47.4%, expand_conv_gemm=19.1%, batchnorm=15.0%, relu6_clamp=12.3%, project_conv_gemm=6.2%
- **Trial 1**: Score of 63.2 corresponds to latency of ~4.31ms on this MBConv benchmark
- **Trial 2**: Trial 2 produced no agent output at all — agent may have gotten stuck on planning without executing
- **Trial 2**: When retrying after a failed trial, always start by restoring the last known working solution and confirming it works before attempting new optimizations
- **Trial 3**: Agent got stuck with no output in trials 2 and 3 — need extremely explicit step-by-step instructions with exact code to copy
- **Trial 3**: For MBConv optimization, try channels_last memory format (NHWC) which is typically faster for conv-heavy workloads on AMD GPUs
- **Trial 3**: torch.compile mode='max-autotune' may produce better kernels than mode='default' by trying more Triton autotuning configurations
- **Trial 3**: BN folding into conv weights can eliminate BN kernels entirely (15% of runtime) for inference
- **Trial 4**: Agent has gotten stuck with no output for 3 consecutive trials on MBConv optimization — needs complete copy-paste code
- **Trial 4**: When agent produces no output repeatedly, provide the entire solution file content rather than incremental instructions
- **Trial 5**: Agent has failed to produce output for 4 consecutive trials (2-5) on MBConv optimization — likely getting stuck in planning/analysis loops
- **Trial 5**: When agent repeatedly produces no output, provide shell commands (cat > file << EOF) rather than Python code blocks to ensure execution
- **Trial 5**: channels_last memory format + torch.compile(max-autotune) + eval() mode is the next optimization to try after torch.compile(default) achieved 63.2 score
