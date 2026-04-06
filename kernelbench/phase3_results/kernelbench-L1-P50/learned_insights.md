# Learned Insights

- **Trial 1**: torch.compile fails on ROCm Triton with 'ttg.async_copy_global_to_local legalization failure' MLIR error for conv2d workloads
- **Trial 1**: Manual Triton conv2d kernels that work for small batch sizes but fail at batch_size=256 likely have int32 overflow in index computation — total elements ~71M requires int64 indexing
- **Trial 1**: F.unfold-based im2col for conv2d(batch=256, in=3, H=224, W=224, out=96, k=11, s=4) creates a ~2GB intermediate tensor that is 10x slower than MIOpen
- **Trial 1**: MIOpen convolution kernel (miopen_convolution) takes 91.23% of baseline time at 1.24ms total — very hard to beat with naive Triton
- **Trial 1**: Bias add after conv2d is a separate kernel taking 8.77% (~116us) — fusion opportunity
- **Trial 1**: For KernelBench scoring, wrapping torch.nn.functional.conv2d gives score=60.0 as the baseline
- **Trial 2**: Trial 2 produced no output - agent likely got stuck in compilation or debugging loop without producing a valid generated_kernel.py
- **Trial 2**: For KernelBench conv2d on MI355X, the pragmatic approach is MIOpen conv + Triton bias fusion since MIOpen is vendor-optimized and bias is 8.77% of runtime
- **Trial 2**: When supervisor_tightens stage has no metric, must retry to get a working baseline before tightening
- **Trial 3**: Agent has failed 2 consecutive trials with no output on conv2d optimization - needs extremely concrete copy-paste solutions
- **Trial 3**: For KernelBench conv2d on MI355X, pragmatic strategy is MIOpen conv (via F.conv2d) + Triton bias fusion since MIOpen handles 91% of runtime
- **Trial 3**: When agent produces no output repeatedly, give near-complete code in hints rather than algorithmic guidance
- **Trial 4**: Agent has failed 3 consecutive trials on conv2d optimization - needs complete copy-paste code, not algorithmic guidance
- **Trial 4**: For KernelBench conv2d (problem 50), channels_last memory format may speed up MIOpen convolution on MI355X
- **Trial 4**: Splitting F.conv2d(bias=None) + Triton bias_add_kernel avoids the separate aten::add_ bias kernel (8.77% of runtime)
- **Trial 4**: When agent produces no output repeatedly, provide TWO solutions: an ambitious one and a safe fallback that just wraps the original
- **Trial 5**: Agent has failed 4 consecutive trials with no output on conv2d optimization - fundamental execution issue, not a code quality issue
- **Trial 5**: When agent is stuck in a no-output loop, reduce the task to a single copy-paste shell command with heredoc
