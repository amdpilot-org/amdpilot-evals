# Learned Insights

- **Trial 1**: KernelBench scoring: SCORE = 50 + 50 * min(speedup/5, 1.0), so 1.0x speedup = 60.0, need >1.0x for higher scores
- **Trial 1**: 3D transposed convolution on AMD ROCm uses MIOpen GEMM (ck::tensor_operation::device::kernel_grouped_conv_bwd_data) which is 81.7% of compute — very hard to beat
- **Trial 1**: batched_transpose_32x16_dword and batched_transpose_16x32_dword account for 18.3% — these are data layout conversions that channels_last_3d format might eliminate
- **Trial 1**: Triton only supports 3D grids (program_id 0,1,2), making full 5D conv kernels impractical without complex tiling
- **Trial 1**: torch.compile mode=max-autotune on AMD ROCm can cause CUDAGraph/backwards issues and performance regressions for conv operations
- **Trial 1**: Writing a full Triton conv_transpose3d from scratch is not viable due to Triton control flow limitations (no continue, 3D grid only)
- **Trial 2**: Trial 2 produced no output — agent may have gotten stuck on planning without executing
- **Trial 2**: channels_last_3d memory format is the key lever for eliminating the 18.3% transpose overhead in MIOpen conv_transpose3d
- **Trial 3**: Agent got stuck on trials 2 and 3 with no output — need extremely specific code-level instructions
- **Trial 3**: channels_last_3d format conversion should be done in __init__ not forward to avoid per-call overhead
- **Trial 3**: With ~29 min remaining, must provide near-complete code rather than high-level guidance
- **Trial 4**: Agent has been stuck for 3 consecutive trials producing no output on 3D transposed conv optimization — needs complete code, not guidance
- **Trial 4**: channels_last_3d memory format is the primary lever to eliminate batched_transpose kernels (18.3% of runtime)
- **Trial 4**: Weight conversion to channels_last_3d should happen in __init__, input conversion in forward
- **Trial 5**: Agent has been stuck for 4 consecutive trials on 3D transposed conv optimization — producing zero output each time
- **Trial 5**: When agent is stuck, provide shell commands to run verbatim rather than code to write — use cat with heredoc
- **Trial 5**: With <30 minutes remaining and 4 consecutive failures, consider stopping rather than retrying
