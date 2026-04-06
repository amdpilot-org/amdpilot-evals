# Learned Insights

- **Trial 1**: For Matmul+AvgPool patterns, avgpool can be algebraically fused into the matmul weights since both are linear operations. W_pooled = W.reshape(out//k, k, in).mean(dim=1) reduces the GEMM size by factor k.
- **Trial 1**: rocBLAS GEMM via torch.nn.Linear dominates at 96.4% of runtime for this problem size (8192x8192). Naive Triton matmul is 163x slower.
- **Trial 1**: tl.math.tanh is unavailable on ROCm - use exp-based tanh: exp_2x = exp(2*x); tanh = (exp_2x - 1)/(exp_2x + 1)
- **Trial 1**: Fusing only the non-GEMM tail (pool+gelu+scale+max = 3.5% of runtime) gives at most 1.01x speedup
- **Trial 2**: For Matmul+AvgPool patterns where pool_kernel_size=16, algebraic fusion reduces GEMM from (out_features x in_features) to (out_features/16 x in_features), giving 16x compute reduction
- **Trial 2**: Pre-computing W_pooled in __init__ is safe because both Linear and AvgPool1d are linear operations that commute
- **Trial 2**: Agent produced no output in trial 2 - may need explicit instructions to avoid environment issues and produce the file directly
- **Trial 3**: Agent failed silently in trials 2 and 3 - provide complete ready-to-write code to avoid execution issues
- **Trial 3**: Algebraic fusion of AvgPool into Linear weights: W_pooled = W.reshape(out//k, k, in).mean(dim=1), b_pooled = b.reshape(out//k, k).mean(dim=1)
- **Trial 3**: This reduces GEMM from (8192, 8192) to (512, 8192) - a 16x compute reduction since pool_kernel_size=16
- **Trial 4**: Agent has failed silently 3 consecutive times - provide complete code inline rather than algorithmic descriptions
- **Trial 4**: Must keep self.matmul = nn.Linear(in_features, out_features) so state_dict loading from reference model works, then cache fused weights on first forward call
- **Trial 4**: Algebraic fusion: W_pooled = W.reshape(out//k, k, in).mean(1) reduces GEMM from (8192,8192) to (512,8192), b_pooled = b.reshape(out//k, k).mean(1)
- **Trial 5**: Agent has silently failed 4 consecutive times — must provide verbatim file content and extremely simple instructions
- **Trial 5**: For BLOCK_SIZE with N=512 (8192/16), use BLOCK_SIZE=512 to avoid wasting wavefronts
- **Trial 5**: torch.nn.functional.linear with pre-fused weights avoids needing a separate nn.Linear for the reduced GEMM
