# Learned Insights

- **Trial 1**: torch.set_float32_matmul_precision('high') improves GEMM performance on MI355X for both reference and custom implementations
- **Trial 1**: tl.math.tanh is unavailable on ROCm Triton - must use manual formula: (exp(2x) - 1) / (exp(2x) + 1)
- **Trial 1**: For GEMM-dominated workloads (like 2048x8192 @ 8192x8192), fusing only elementwise ops gives minimal speedup (~6.5%); real gains require GEMM epilogue fusion
- **Trial 1**: BLOCK_SIZE must be multiple of 64 for CDNA4 wavefront alignment on MI355X
- **Trial 1**: torch.compile conflicts with hand-written Triton kernels causing correctness issues - use one approach or the other, not both
- **Trial 2**: Trial 2 produced no output - need to investigate if generated_kernel.py was corrupted or if there was an environment issue
- **Trial 2**: For GEMM-dominated workloads, the only way to get significant speedup is fusing GEMM with epilogue operations to avoid extra global memory traffic
- **Trial 2**: For 2048x8192 @ 8192x8192 matmul on MI355X, try tile sizes BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 with num_warps=8
- **Trial 3**: Two consecutive trials with no output suggest the agent is crashing during code generation or the generated_kernel.py file is in a broken state — always verify file state first
- **Trial 3**: For KernelBench problems, torch.compile(mode='max-autotune') on pure PyTorch forward pass is a safe baseline optimization that avoids hand-written kernel bugs
- **Trial 3**: Do NOT mix torch.compile with hand-written @triton.jit kernels — they conflict and cause correctness issues
- **Trial 4**: Three consecutive no-output failures suggest the generated_kernel.py is in a broken state - always read and verify the file before making changes
- **Trial 4**: For GEMM-dominated problems on MI355X, torch.compile(mode='max-autotune') may achieve automatic epilogue fusion which is hard to do manually
- **Trial 4**: When the agent crashes repeatedly, the recovery priority is: (1) verify file state, (2) run benchmark to confirm working, (3) make incremental changes
- **Trial 5**: Four consecutive no-output failures indicate the agent is corrupting generated_kernel.py and not recovering — explicit file content must be provided in hints
- **Trial 5**: For stability, always cat the current file state before making any changes to diagnose corruption
