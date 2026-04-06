# Learned Insights

- **Trial 1**: torch.compile(mode='default') gives 1.42x speedup on SqueezeNet (16.3ms -> 11.5ms) on MI355X
- **Trial 1**: Manual Triton conv2d kernels fail on ROCm gfx950 with 'failed to legalize operation ttg.async_copy_global_to_local' MLIR error
- **Trial 1**: SqueezeNet profiling breakdown: convolutions 39.8%, elementwise 24.4%, pooling 13.3%, other 22.5%
- **Trial 1**: MIOpen assembly convolutions (miopenSp3AsmConv) dominate compute — don't try to replace them with Triton
- **Trial 1**: Standalone Triton kernel launches on ROCm can hit 'invalid argument' errors — test kernels incrementally
- **Trial 2**: Trial 2 produced no output — possibly the agent timed out or hit an error before running anything
- **Trial 2**: torch.compile(mode='default') is the proven baseline achieving score 64.20 on SqueezeNet
- **Trial 3**: Trial 2 and 3 both produced no output — agent may be hitting a silent crash or timeout before benchmark execution
- **Trial 3**: When retrying after no-output failures, the agent must verify the generated_kernel.py file is valid before attempting optimizations
- **Trial 4**: Trials 2-4 all produced no output on SqueezeNet optimization — agent may be attempting complex changes that crash before benchmark execution
- **Trial 4**: When agent produces no output repeatedly, give step-by-step instructions starting with verifying existing solution works before making changes
- **Trial 4**: For torch.compile optimization beyond mode='default', try inductor_config settings: conv_1x1_as_mm=True, coordinate_descent_tuning=True, max_autotune=True
- **Trial 5**: Trials 2-5 all produced zero output on SqueezeNet — the agent consistently crashes when attempting complex changes beyond the initial torch.compile solution
- **Trial 5**: When 4+ consecutive trials fail with no output, give extremely prescriptive step-by-step instructions with verification checkpoints
- **Trial 5**: Always verify the existing working solution before attempting changes — run benchmark first to establish the solution still works
