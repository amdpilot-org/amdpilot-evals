# Learned Insights

- **Trial 1**: For KernelBench L2P55: matmul (128x32768 @ 32768x32768) is 98% of runtime at ~1.24ms
- **Trial 1**: Custom Triton matmul was 2x slower than rocBLAS on MI355X for this problem size — use torch.mm/F.linear instead
- **Trial 1**: torch.compile mode=max-autotune causes CUDAGraph issues on ROCm — use mode=default
- **Trial 1**: torch.compile mode=default with tensor ops (replacing nn.MaxPool1d) gives best fusion via Inductor
- **Trial 1**: Score 60.1 corresponds to ~1.008x speedup — marginal improvement only
- **Trial 1**: For large GEMM-dominated workloads, switching to FP16/BF16 is the highest-leverage optimization
- **Trial 2**: Trial 2 produced no agent output — possible stall or environment issue, need explicit starting instructions
- **Trial 2**: For KernelBench scoring, a score of 60.10 corresponds to only ~1.008x speedup over baseline — significant room for improvement via dtype optimization
- **Trial 3**: Agent stalled on trials 2 and 3 with no output — needs extremely explicit step-by-step instructions with complete code
- **Trial 3**: For GEMM-dominated workloads on MI355X, FP16 matmul should give ~2x speedup since GEMM is 98% of runtime
- **Trial 3**: When agent stalls, provide the complete replacement code rather than abstract guidance
- **Trial 4**: Agent has stalled on trials 2-4 with no output — must provide complete copy-paste commands
- **Trial 4**: FP16 GEMM is the highest-leverage optimization for this GEMM-dominated workload (98% runtime)
- **Trial 4**: Need to watch for correctness tolerance issues when switching to FP16 matmul
- **Trial 5**: Agent has stalled 4 consecutive trials (2-5) with no output despite increasingly explicit instructions
- **Trial 5**: FP16 linear layer should give ~2x GEMM speedup since GEMM is 98% of runtime for 128x32768 @ 32768x32768
- **Trial 5**: When agent repeatedly stalls, provide complete heredoc file write + single benchmark command as the only instructions
