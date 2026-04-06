# Learned Insights

- **Trial 1**: KernelBench P31: Conv2d dominates at 85.5% (MIOpen assembly kernel), fused elementwise ops are only 14.5%
- **Trial 1**: Folding conv bias into fused kernel eliminates a separate aten::add_ kernel launch (~0.4ms savings)
- **Trial 1**: Conv bias must be added BEFORE min operation to maintain correctness
- **Trial 1**: BLOCK_SIZE=4096 works well for elementwise Triton kernels on MI355X
- **Trial 1**: Original flattened index calculation is faster than alternative approaches for channel indexing
- **Trial 1**: Score formula appears to be: score = (reference_time / optimized_time) * (some_factor) — score 63.1 with 1.31x speedup
- **Trial 2**: Trial 2 of KernelBench P31 produced no output - agent may have gotten stuck without running the benchmark
- **Trial 2**: With conv at 85.5% of time, channels_last memory format and torch.compile on conv layer are the main remaining levers
- **Trial 3**: Trial 3 of KernelBench P31 also produced no output - agent is getting stuck without running benchmark on optimization stages
- **Trial 3**: Agent needs extremely explicit step-by-step instructions to avoid getting stuck in analysis paralysis when conv dominates
- **Trial 4**: KernelBench P31: Agent gets stuck in analysis paralysis on optimization stages when conv dominates - needs extremely explicit step-by-step instructions with exact code to paste
- **Trial 4**: After 3 consecutive no-output trials, the agent needs instructions that minimize decision-making and just say 'paste this code, run this command'
- **Trial 5**: KernelBench P31: Agent has failed to produce output 4 times in a row on optimization stages - needs absolute minimum decision-making in instructions
- **Trial 5**: channels_last memory format is the primary remaining optimization lever when MIOpen conv dominates at 85.5%
