# Learned Insights

- **Trial 1**: PyTorch F.gelu() uses exact GELU with erf (not tanh approximation). tl.math.erf IS available on ROCm Triton and must be used for correctness.
- **Trial 1**: Channel index calculation for (N,C,D,H,W) layout: (offset // channel_stride) % num_channels
- **Trial 1**: BLOCK_SIZE should be multiple of 64 for CDNA4 wavefront alignment on MI355X
- **Trial 1**: Conv3d dominates runtime in this problem — post-conv fusion (LeakyReLU+Sum+Clamp+GELU) gives 1.28x speedup but further gains require optimizing conv3d itself
- **Trial 1**: Tanh-based GELU approximation fails with max_difference=0.000153 exceeding 0.0001 tolerance
- **Trial 2**: Conv3d dominates runtime in this problem — post-conv fusion (LeakyReLU+Sum+Clamp+GELU) gives 1.28x speedup but further gains require optimizing conv3d itself
- **Trial 2**: torch.compile(mode='max-autotune') on Conv3d is a promising next optimization to try
- **Trial 2**: Agent may fail silently if not given explicit step-by-step instructions — always start with verifying existing solution works
- **Trial 3**: Agent failed silently in trials 2 and 3 — needs extremely explicit step-by-step instructions with verification checkpoints
- **Trial 3**: Conv3d dominates runtime in this problem — post-conv fusion (LeakyReLU+Sum+Clamp+GELU) gives 1.28x speedup but further gains require optimizing conv3d itself
- **Trial 3**: torch.compile(mode='max-autotune') on Conv3d is a promising next optimization to try
- **Trial 4**: Agent has failed silently 3 consecutive times (trials 2-4) — needs copy-paste-ready instructions
- **Trial 4**: Conv3d dominates runtime in this problem — post-conv fusion (LeakyReLU+Sum+Clamp+GELU) gives 1.28x speedup but further gains require optimizing conv3d itself
- **Trial 4**: torch.compile(mode='max-autotune') on Conv3d and torch.backends.cudnn.benchmark=True are untested optimization opportunities
- **Trial 5**: Agent has failed silently 4 consecutive times (trials 2-5) — likely getting stuck on complex code generation without executing anything
- **Trial 5**: Conv3d dominates runtime in this problem — post-conv fusion (LeakyReLU+Sum+Clamp+GELU) gives 1.28x speedup but further gains require optimizing conv3d itself
- **Trial 5**: torch.compile(mode='max-autotune') on Conv3d and torch.backends.cudnn.benchmark=True are still untested optimization opportunities after 5 trials
