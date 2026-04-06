# Learned Insights

- **Trial 1**: For Conv2d+BN+Scaling fusion on MI355X, torch.compile(mode='default') achieves 1.74x speedup (0.97ms vs 1.69ms baseline)
- **Trial 1**: Manual Triton kernels for elementwise scaling and fused BN+scaling had higher overhead than PyTorch native ops for batch_size=128, out_channels=64, 126x126 spatial dims
- **Trial 1**: torch.compile without explicit mode can hit async_copy_global_to_local Triton compilation errors on ROCm
- **Trial 1**: ModelNew must have identical architecture and parameter initialization order as reference Model for weight loading to produce correct results
- **Trial 2**: Conv-BN fusion in eval mode eliminates BatchNorm entirely by folding affine params into conv weights, removing a full tensor operation
- **Trial 2**: Score metric is higher=better; score 67.50 corresponds to ~0.97ms execution time with torch.compile(mode='default')
- **Trial 2**: When trial produces no output, the agent likely crashed during setup - always verify the solution file exists before optimizing
- **Trial 3**: Two consecutive trials crashed with no output - agent may be running pip install or other setup that breaks the environment
- **Trial 3**: Conv-BN fusion eliminates BatchNorm by folding affine parameters into conv weights, reducing from 3 ops to 1 conv op
- **Trial 3**: Always verify the solution file exists and benchmark runs before attempting optimization
- **Trial 4**: Agent crashed 3 consecutive trials (2,3,4) with no output - likely environment corruption from pip install or similar setup commands
- **Trial 4**: For Conv2d+BN+Scaling on MI355X, torch.compile(mode='default') gives 1.74x speedup; manual Triton kernels are slower
- **Trial 4**: Conv-BN fusion (folding BN affine params into conv weights) + torch.compile should eliminate BN overhead entirely
- **Trial 5**: Agent crashed 4 consecutive trials (2-5) with no output - almost certainly due to environment-corrupting commands like pip install
- **Trial 5**: For KernelBench problems, the generated_kernel.py must define ModelNew with the same __init__ signature as the reference Model
- **Trial 5**: Conv-BN fusion folding BN affine params into conv weights eliminates the BN forward pass entirely, reducing to conv + scale
