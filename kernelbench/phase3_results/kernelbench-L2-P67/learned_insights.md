# Learned Insights

- **Trial 1**: KernelBench L2P67: tl.math.tanh is unavailable on ROCm Triton; must use tl.math.erf for GELU to match PyTorch's exact formula (tanh approx gives max_diff=0.000128 > 1e-4 tolerance)
- **Trial 1**: KernelBench L2P67: Fusing Conv2d bias addition into the post-conv Triton kernel saved 0.77ms (23% of total runtime) by eliminating a separate bias add kernel and its memory round-trip
- **Trial 1**: KernelBench L2P67: BLOCK_SIZE=4096 with 16 iterations over 65536 spatial elements gives best performance; BLOCK_SIZE=8192 caused slight regression due to register pressure
- **Trial 1**: KernelBench L2P67: Conv2d (miopenSp3AsmConv) dominates at 73% of runtime (~1.56ms) and is the primary remaining bottleneck
- **Trial 1**: KernelBench L2P67: Score formula is likely speedup-based: score=65.9 corresponds to 1.59x speedup (2.13ms vs 3.37ms baseline)
- **Trial 2**: KernelBench L2P67: Trial 2 produced no output - agent may have gotten stuck on complex rewrites. Incremental changes from a working baseline are safer.
- **Trial 2**: KernelBench L2P67: channels_last memory format is a key optimization to try for Conv2d-dominated workloads on AMD GPUs
- **Trial 3**: KernelBench L2P67: Agent crashed/timed out in trials 2 and 3 with no output - must give very specific step-by-step instructions and enforce incremental changes
- **Trial 3**: KernelBench L2P67: channels_last memory format and torch.compile on Conv2d are untried optimizations for the 73% Conv2d bottleneck
- **Trial 4**: KernelBench L2P67: Agent has crashed/timed out 3 consecutive times (trials 2-4) with no output when attempting complex optimizations - must enforce extremely minimal changes
- **Trial 4**: KernelBench L2P67: channels_last memory format and torch.compile on Conv2d remain untried after 4 trials despite being identified as key optimizations for the 73% Conv2d bottleneck
- **Trial 5**: KernelBench L2P67: Agent has crashed/timed out 4 consecutive times (trials 2-5) - must enforce absolute minimal changes and step-by-step verification
- **Trial 5**: KernelBench L2P67: channels_last memory format and torch.compile on Conv2d remain untried after 5 trials despite being the most promising optimizations for the 73% Conv2d bottleneck
