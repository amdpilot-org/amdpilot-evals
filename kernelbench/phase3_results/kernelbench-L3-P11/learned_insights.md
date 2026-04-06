# Learned Insights

- **Trial 1**: VGG16 on MI355X: convolutions are 78.6% of GPU time, dominated by MIOpen assembly kernels (miopenSp3AsmConv 53.6%, igemm_fwd_gtcx35_nhwc 25.0%)
- **Trial 1**: torch.compile(mode='default') gives ~7.8% speedup on VGG16 by fusing ReLU into convolutions and autotuning linear layer GEMM kernels
- **Trial 1**: Custom Triton conv kernels fail on ROCm with 'failed to legalize operation ttg.async_copy_global_to_local'
- **Trial 1**: Custom Triton linear+ReLU fusion kernel was 1.77x SLOWER than hipBLAS (6.14ms vs 3.46ms)
- **Trial 1**: Baseline VGG16: reference 3.46ms, score 50.0; with torch.compile: 3.19ms, score 60.8
- **Trial 2**: Trial 2 agent produced no output - likely crashed during compilation or optimization attempt. Need explicit recovery instructions.
- **Trial 2**: VGG16 score metric is higher-is-better: baseline 50.0, torch.compile default gives 60.7
- **Trial 2**: Convolutions are 78.6% of GPU time on VGG16 - MIOpen assembly kernels handle these, not Triton
- **Trial 3**: VGG16 optimization: agent crashed in trials 2 and 3 with no output - need very explicit step-by-step instructions with frequent checkpoints
- **Trial 3**: For VGG16 on MI355X, promising untried optimizations: cudnn.benchmark=True, channels_last memory format, torch.compile mode='max-autotune'
- **Trial 4**: VGG16 optimization: agent crashed in trials 2, 3, and 4 with no output - likely attempting complex changes that fail silently
- **Trial 4**: For crashed trials, provide explicit cat/run/modify/run workflow with no ambiguity
- **Trial 4**: channels_last memory format and cudnn.benchmark=True are untried optimizations for VGG16 on MI355X
- **Trial 4**: torch.compile mode='max-autotune' is untried - may give better kernel selection than mode='default'
- **Trial 5**: VGG16 optimization: agent has crashed 4 consecutive trials (2-5) with no output - must provide complete copy-pasteable solution code
- **Trial 5**: For reliability, avoid any custom Triton kernel attempts on VGG16 - MIOpen handles convolutions better
- **Trial 5**: Next untried combo: channels_last + cudnn.benchmark + torch.compile(mode='max-autotune')
