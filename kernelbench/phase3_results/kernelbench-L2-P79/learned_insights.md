# Learned Insights

- **Trial 1**: KernelBench L2P79: Conv3d dominates at 59.4% (MIOpen), cannot be optimized with Triton
- **Trial 1**: KernelBench L2P79: Reducing Triton grid from 1.6M to ~6400 programs (BLOCK_SPATIAL=256, loop over channels) was critical for performance
- **Trial 1**: KernelBench L2P79: InstanceNorm's internal batch_norm kernels contribute 8.4% + transpose ops contribute 11.4% - fusing these into Triton would save ~20%
- **Trial 1**: KernelBench L2P79: Two-pass mean/variance in Triton had precision issues - use Welford's algorithm with float32 accumulation
- **Trial 1**: KernelBench L2P79: Baseline PyTorch 0.816ms, current best 0.745ms (1.095x speedup, score 60.90)
- **Trial 2**: KernelBench L2P79: Trial 2 failed with no output - agent may have stalled on implementation complexity
- **Trial 2**: KernelBench L2P79: Two-kernel approach needed: (1) reduce mean/var per (b,c), (2) fused normalize+clamp+mult+max per spatial position
- **Trial 3**: KernelBench L2P79: Agent has stalled twice with no output on InstanceNorm fusion - need extremely specific guidance or simpler approach
- **Trial 3**: KernelBench L2P79: Fallback if InstanceNorm fusion is too hard: try torch.compile on forward(), or increase BLOCK_SPATIAL, or just submit existing kernel
- **Trial 4**: KernelBench L2P79: Agent stalls repeatedly when attempting InstanceNorm fusion in Triton - this approach is too complex
- **Trial 4**: KernelBench L2P79: When agent produces no output 3x, give extremely constrained instructions with a single concrete change to try
- **Trial 4**: KernelBench L2P79: Fallback strategy: use torch.compile for conv+instancenorm while keeping existing Triton kernel for clamp+mult+max
- **Trial 5**: KernelBench L2P79: Agent stalls 4+ times on InstanceNorm Triton fusion - this approach is definitively too complex for the agent
- **Trial 5**: KernelBench L2P79: When agent stalls repeatedly, provide copy-paste-ready instructions with explicit fallback to submitting existing working code
- **Trial 5**: KernelBench L2P79: Always ensure the agent can fall back to the last working solution rather than producing no output
