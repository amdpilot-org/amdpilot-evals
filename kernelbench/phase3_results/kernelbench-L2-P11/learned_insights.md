# Learned Insights

- **Trial 1**: KernelBench score=61 corresponds to 2.61ms optimized vs 2.90ms reference (1.11x speedup) for Level 2 Problem 11
- **Trial 1**: torch.compile(mode='max-autotune') causes 4x slowdown on ROCm MI300X - always use mode='default'
- **Trial 1**: For ConvTranspose2d+BN+Tanh+MaxPool+GroupNorm pipeline, transposed conv GEMM (igemm_bwd_gtcx35) dominates at 78.2% of runtime
- **Trial 1**: Custom Triton kernels for elementwise ops had correctness issues likely due to parameter initialization order mismatch between Model and ModelNew
- **Trial 1**: Environment vars GPU_MAX_HW_QUEUES=2, HIP_FORCE_DEV_KERNARG=1, PYTORCH_TUNABLEOP_ENABLED=1, TORCH_BLAS_PREFER_HIPBLASLT=1 provide marginal gains on ROCm
- **Trial 1**: Unset TORCHINDUCTOR_MAX_AUTOTUNE when using torch.compile on ROCm to avoid performance regression
- **Trial 2**: Trial 2 produced no output - agent may have gotten stuck on complex custom kernel implementation without running benchmark
- **Trial 2**: For KernelBench problems, always verify the current solution still works before attempting modifications
- **Trial 3**: Agent has failed 2 consecutive trials with no output on this stage - likely getting stuck on complex implementations without running benchmark
- **Trial 3**: For KernelBench, always run the benchmark after each small change to avoid wasting time on broken implementations
- **Trial 3**: When fusing BN+Tanh for inference, must use running_mean/running_var (not batch stats) since test harness calls model.eval()
- **Trial 4**: Agent repeatedly fails when attempting complex multi-kernel fusion — must enforce single-change-then-benchmark discipline
- **Trial 4**: 3 consecutive no-output failures indicate the agent is spending all time coding without running anything
- **Trial 4**: Fusing tanh+maxpool is the simplest possible custom kernel since neither op has learned parameters
- **Trial 5**: Agent has failed 4 consecutive trials on this problem by attempting complex implementations without running benchmarks — must enforce run-first discipline
- **Trial 5**: The working solution from trial 1 uses torch.compile(mode='default') wrapping the full model for a score of 61
- **Trial 5**: For this problem, extracting the forward pass into a standalone @torch.compile function may give better fusion opportunities than compiling the whole module
