# Learned Insights

- **Trial 1**: KernelBench L2P5: ConvTranspose2d (MIOpen) dominates at 65.6% of runtime, hard to replace
- **Trial 1**: KernelBench L2P5: MIOpen's ConvTranspose2d internal bias add is a separate kernel launch (16.8%) that can be eliminated by setting bias=False and fusing into downstream Triton kernel
- **Trial 1**: KernelBench L2P5: BLOCK_SIZE=1024 is optimal for the bias_sub_tanh kernel; 256 is slower, 2048/4096 show no improvement
- **Trial 1**: On ROCm Triton, tl.math.tanh is unavailable; use manual implementation: (exp(2x)-1)/(exp(2x)+1)
- **Trial 1**: BLOCK_SIZE should be multiples of 64 (AMD wavefront size) for optimal performance on MI355X
- **Trial 2**: KernelBench L2P5: Trial 2 produced no output - agent may have gotten stuck reading optimization state without acting
- **Trial 2**: KernelBench L2P5: The unfused ConvTranspose2d bias add (16.8%) is the next optimization target - set bias=False and fuse conv_bias into the Triton kernel
- **Trial 3**: KernelBench L2P5: Agent has gotten stuck with no output in 2 consecutive trials - needs extremely explicit step-by-step instructions with actual code
- **Trial 3**: KernelBench L2P5: The next optimization target is fusing conv bias add (16.8%) into the Triton kernel by setting ConvTranspose2d bias=False
- **Trial 4**: KernelBench L2P5: Agent stuck with no output for 3 consecutive trials - needs complete copy-paste code, not instructions
- **Trial 4**: KernelBench L2P5: When fusing conv bias, must handle weight loading - the test harness copies weights from reference model, so conv_bias param name must match or be manually loaded
- **Trial 5**: KernelBench L2P5: Agent stuck for 4 trials with no output - likely overwhelmed by optimization state reading; needs direct executable commands
- **Trial 5**: KernelBench L2P5: When providing code to stuck agents, use cat heredoc syntax to write files directly rather than asking agent to edit
